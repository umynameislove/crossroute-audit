"""BLIP-2 adapter for the clean model-forward path.

BLIP-2 exposes a Q-Former with explicit cross-attention, which gives a clear
intervention surface. This module owns all PyTorch and Transformers imports so
the audit core remains model-agnostic.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Iterator

from PIL import Image
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import yaml

from ..instrumentation.hooks import ActivationHook, managed_forward_hook
from ..interventions.ablation import group_positions, run_ablation
from ..interventions.patching import run_activation_patching
from .base import ForwardOutput, ModelAdapter, TokenGroups


LOGGER = logging.getLogger(__name__)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
_SUPPORTED_TARGET_POLICIES = {
    "exact_token",
    "first_generated_token",
    "selected_candidate_logit",
}
_SUPPORTED_INTERVENTION_MODES = {"ablate", "mask", "patch", "shuffle", "noop"}
_SUPPORTED_INTERVENTION_GROUPS = {"image", "text", "negative_control"}
_TARGET_TOKEN_ID_KEY = "_crossroute_target_token_id"
_CORRUPT_INPUTS_KEY = "_crossroute_corrupt_inputs"


class BLIP2Adapter(ModelAdapter):
    """Adapter for ``Salesforce/blip2-flan-t5-xl``.

    The checkpoint is loaded lazily so importing the package and running
    CPU-only unit tests never downloads model weights.
    """

    name = "blip2"

    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.seed = self._read_seed()
        self.processor: Blip2Processor | None = None
        self.model: Blip2ForConditionalGeneration | None = None

    def prepare_inputs(self, image, question):
        """Prepare one deterministic image-question example.

        ``image`` may be a filesystem path, a ``PIL.Image.Image``, or ``None``
        (which zeroes the pixel values for a no-image baseline). Images are
        converted to RGB and copied before processing so caller-owned image
        objects are not mutated. Integer token tensors stay integer while
        floating image tensors are moved to the model device and dtype.
        """
        self._ensure_loaded()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        zero_pixels = image is None
        pil_image = Image.new("RGB", (224, 224), 0) if zero_pixels else self._load_image(image)
        encoded = self.processor(
            images=pil_image,
            text=question.strip(),
            return_tensors="pt",
        )

        prepared = {}
        for key, value in encoded.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    value = value.to(device=self.device, dtype=torch.float16)
                else:
                    value = value.to(device=self.device)
            prepared[key] = value

        if zero_pixels and "pixel_values" in prepared:
            # True "no image" baseline: zero the vision input.
            prepared["pixel_values"] = torch.zeros_like(prepared["pixel_values"])

        LOGGER.debug(
            "Prepared BLIP-2 inputs: %s",
            {key: list(value.shape) for key, value in prepared.items() if torch.is_tensor(value)},
        )
        return prepared

    def forward(self, inputs, capture: bool = False) -> ForwardOutput:
        """Run a clean first-step forward pass.

        The returned ``target_logit`` is the maximum logit at the first decode
        step. Use :meth:`get_target_logit` when a specific target policy is
        required. With ``capture=True``, hidden states and attentions are
        returned as per-layer scalar/shape summaries; raw tensors are not kept
        in ``ForwardOutput``.
        """
        outputs, first_step_logits = self._run_first_decode_step(inputs, capture=capture)
        top_logit, top_token_id = first_step_logits.max(dim=-1)
        token_id = int(top_token_id[0].item())

        hidden_states = self._summarize_hidden_states(outputs) if capture else None
        attentions = self._summarize_attentions(outputs) if capture else None
        meta = {
            "model_name": self.model_name,
            "device": str(self.device),
            "seed": self.seed,
            "capture": capture,
            "first_step_logits_shape": list(first_step_logits.shape),
            "predicted_token_id": token_id,
            "predicted_token": self.processor.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
            ).strip(),
            "input_shapes": {
                key: list(value.shape)
                for key, value in inputs.items()
                if torch.is_tensor(value)
            },
        }
        return ForwardOutput(
            target_logit=float(top_logit[0].item()),
            hidden_states=hidden_states,
            attentions=attentions,
            meta=meta,
        )

    def get_token_groups(self, inputs) -> TokenGroups:
        """Return first-step token indices for the language-model sequence.

        Transformers 4.57+ prepends ``num_query_tokens`` image placeholders
        directly to ``input_ids``. BLIP-2 replaces those placeholder embeddings
        with projected Q-Former outputs before the language model. ``image``
        contains the placeholder positions, ``text`` contains the remaining
        non-padding prompt positions, ``fusion`` covers all valid encoder
        positions, and ``answer`` is decoder position zero.
        """
        self._ensure_loaded()
        input_ids = inputs.get("input_ids")
        if input_ids is None or not torch.is_tensor(input_ids):
            raise ValueError("inputs must contain an input_ids tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("get_token_groups currently supports one sample at a time")

        num_query_tokens = int(self.model.config.num_query_tokens)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            valid_positions = attention_mask[0].nonzero(as_tuple=False).flatten().tolist()
        else:
            valid_positions = list(range(int(input_ids.shape[1])))
        if len(valid_positions) < num_query_tokens:
            raise ValueError(
                f"input sequence has {len(valid_positions)} valid positions, fewer than "
                f"num_query_tokens={num_query_tokens}"
            )

        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.model.config, "image_token_index", None)
        if image_token_id is None:
            image_positions = valid_positions[:num_query_tokens]
        else:
            image_positions = [
                position
                for position in valid_positions
                if int(input_ids[0, position].item()) == int(image_token_id)
            ]
            if len(image_positions) != num_query_tokens:
                raise ValueError(
                    "processor/model image placeholder count mismatch: "
                    f"expected {num_query_tokens}, got {len(image_positions)}"
                )

        image_position_set = set(image_positions)
        text_positions = [
            position for position in valid_positions if position not in image_position_set
        ]

        groups = TokenGroups(
            image=image_positions,
            text=text_positions,
            fusion=valid_positions,
            answer=[0],
        )
        LOGGER.debug(
            "BLIP-2 token groups: processor_input_shape=%s valid_length=%d "
            "image=%d text=%d fusion=%d answer=%d",
            list(input_ids.shape),
            len(valid_positions),
            len(groups.image),
            len(groups.text),
            len(groups.fusion),
            len(groups.answer),
        )
        return groups

    def get_layer_count(self) -> int:
        """Return Q-Former encoder layers, preserving the Phase 2 contract.

        Phase 3 routing proxies and interventions use a separate zero-based
        language-model encoder layer index validated against
        ``language_model.encoder.block``.
        """
        self._ensure_loaded()
        return int(self.model.config.qformer_config.num_hidden_layers)

    def get_intervention_layer_count(self) -> int:
        """Number of language-model encoder layers — the axis used by
        ``intervene``, ``get_routing_proxy``, and per-layer causal metrics.
        Distinct from ``get_layer_count`` (Q-Former layers).
        """
        self._ensure_loaded()
        return len(self._lm_encoder_layers())

    def prepare_attribution_inputs(
        self,
        inputs,
        target_answer: str,
        policy: str,
    ):
        """Build the differentiable LM-encoder inputs used by layer attribution.

        The returned embedding tensor exactly matches the sequence presented to
        the Flan-T5 encoder during a clean BLIP-2 forward pass: projected
        Q-Former outputs replace image placeholder embeddings and text positions
        retain the language-model token embeddings. The caller uses an all-zero
        tensor of the same shape as the Integrated Gradients baseline.

        This helper intentionally detaches the vision/Q-Former computation.
        Phase-4 attribution is defined at LM-encoder activations, not at pixels
        or Q-Former parameters.
        """
        self._ensure_loaded()
        if self.model.config.use_decoder_only_language_model:
            raise RuntimeError(
                "LM-encoder attribution requires an encoder-decoder BLIP-2 "
                "checkpoint such as Salesforce/blip2-flan-t5-xl"
            )

        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(pixel_values):
            raise ValueError("attribution inputs must contain pixel_values")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("attribution inputs must contain batched input_ids")

        # Resolve and store the exact target token before any baseline forward.
        self.get_target_logit(inputs, target_answer, policy)
        target_token_id = int(inputs[_TARGET_TOKEN_ID_KEY])

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if not torch.is_tensor(attention_mask) or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"pixel_values", "input_ids", "attention_mask", "inputs_embeds"}
        }
        start_token_id = self.model.config.text_config.decoder_start_token_id
        if start_token_id is None:
            start_token_id = self.model.config.text_config.pad_token_id
        if start_token_id is None:
            raise ValueError("decoder start token id and pad token id are both undefined")
        model_inputs["decoder_input_ids"] = torch.full(
            (input_ids.shape[0], 1),
            int(start_token_id),
            dtype=torch.long,
            device=input_ids.device,
        )

        captured_embeddings = []

        def capture_encoder_inputs(module, args, kwargs):
            del module
            encoder_inputs = kwargs.get("inputs_embeds")
            if encoder_inputs is None and args:
                encoder_inputs = args[0]
            if not torch.is_tensor(encoder_inputs) or encoder_inputs.ndim != 3:
                raise RuntimeError(
                    "BLIP-2 did not pass [batch, sequence, hidden] embeddings "
                    "to the language-model encoder"
                )
            captured_embeddings.append(encoder_inputs.detach())

        encoder = self.model.language_model.encoder
        handle = encoder.register_forward_pre_hook(
            capture_encoder_inputs,
            with_kwargs=True,
        )
        try:
            # In Transformers 4.57.6, the model's own forward obtains projected
            # Q-Former features from get_image_features(return_dict=True), then
            # scatters them into image-placeholder token embeddings. Capturing
            # the resulting encoder input keeps this path identical even if the
            # helper's public return shape changes across versions.
            with torch.no_grad(), self._capture_settings(False):
                self.model(
                    **model_inputs,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                    use_cache=False,
                )
        finally:
            handle.remove()

        if len(captured_embeddings) != 1:
            raise RuntimeError(
                "expected exactly one language-model encoder call while "
                f"preparing attribution inputs, got {len(captured_embeddings)}"
            )
        encoder_embeddings = captured_embeddings[0].clone()

        if encoder_embeddings.shape[:2] != input_ids.shape:
            raise ValueError(
                "LM-encoder embeddings and input_ids have inconsistent sequence shapes"
            )
        return (
            encoder_embeddings.detach().requires_grad_(True),
            attention_mask.detach(),
            target_token_id,
        )

    def forward_target_logit_from_embeddings(
        self,
        encoder_embeddings,
        attention_mask,
        target_token_id: int,
    ):
        """Return a differentiable first-step target logit for LM embeddings.

        Unlike the clean inference path, this method never enters
        ``torch.inference_mode``. It is reserved for Captum attribution and
        returns one scalar per batch item so gradients can flow from the target
        token through the selected LM-encoder layer.
        """
        self._ensure_loaded()
        if self.model.config.use_decoder_only_language_model:
            raise RuntimeError(
                "LM-encoder attribution requires an encoder-decoder language model"
            )
        if not torch.is_tensor(encoder_embeddings) or encoder_embeddings.ndim != 3:
            raise ValueError(
                "encoder_embeddings must have shape [batch, sequence, hidden]"
            )
        if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if encoder_embeddings.shape[:2] != attention_mask.shape:
            raise ValueError(
                "encoder_embeddings and attention_mask must share batch and sequence dimensions"
            )
        if not isinstance(target_token_id, int):
            raise TypeError("target_token_id must be an integer")

        vocab_size = int(self.model.config.text_config.vocab_size)
        if target_token_id < 0 or target_token_id >= vocab_size:
            raise ValueError(
                f"target token id {target_token_id} is outside vocabulary size {vocab_size}"
            )

        start_token_id = self.model.config.text_config.decoder_start_token_id
        if start_token_id is None:
            start_token_id = self.model.config.text_config.pad_token_id
        if start_token_id is None:
            raise ValueError("decoder start token id and pad token id are both undefined")
        decoder_input_ids = torch.full(
            (encoder_embeddings.shape[0], 1),
            int(start_token_id),
            dtype=torch.long,
            device=encoder_embeddings.device,
        )

        with torch.enable_grad():
            outputs = self.model.language_model(
                inputs_embeds=encoder_embeddings,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
                use_cache=False,
            )
            target_logits = outputs.logits[:, 0, target_token_id]

        if not bool(torch.isfinite(target_logits).all()):
            raise ValueError("differentiable target logit is not finite")
        return target_logits

    @contextmanager
    def attribution_layer_output(self, layer: int):
        """Expose one LM block's primary hidden-state output as a Captum layer.

        T5 blocks return tuples containing hidden states and position-bias
        tensors. Captum should attribute only the primary hidden-state tensor,
        so a temporary identity module is inserted around that tensor. The
        forward hook is always removed, including when attribution raises.
        """
        layer_module = self._lm_encoder_layer(layer)
        tap = torch.nn.Identity()

        def route_hidden_state(module, args, output):
            del module, args
            if torch.is_tensor(output):
                return tap(output)
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                return (tap(output[0]), *output[1:])
            if isinstance(output, list) and output and torch.is_tensor(output[0]):
                return [tap(output[0]), *output[1:]]
            raise TypeError(
                "LM encoder layer output must be a tensor or a sequence whose "
                "first item is a tensor"
            )

        handle = layer_module.register_forward_hook(route_hidden_state)
        try:
            yield tap
        finally:
            handle.remove()

    def get_routing_proxy(self, inputs, layer: int):
        """Return cross-modal self-attention mass at one LM encoder layer.

        ``layer`` is a zero-based index into ``language_model.encoder.block``;
        it is not a Q-Former index and is therefore independent of
        :meth:`get_layer_count`. The scalar averages text-to-image and
        image-to-text attention mass over batch items, heads, and query tokens.
        """
        self._lm_encoder_layer(layer)
        outputs, _ = self._run_first_decode_step(inputs, capture=True)
        language_outputs = getattr(outputs, "language_model_outputs", None)
        attentions = getattr(language_outputs, "encoder_attentions", None)
        if attentions is None or layer >= len(attentions) or attentions[layer] is None:
            raise RuntimeError(
                "language-model encoder attentions were not materialized; "
                "load BLIP-2 with attn_implementation='eager'"
            )

        attention = attentions[layer]
        if attention.ndim != 4:
            raise ValueError(
                "encoder attention must have shape [batch, heads, query, key], "
                f"got {list(attention.shape)}"
            )
        groups = self.get_token_groups(inputs)
        image_index = self._position_tensor(groups.image, attention.shape[-1], attention.device)
        text_index = self._position_tensor(groups.text, attention.shape[-1], attention.device)

        text_to_image = (
            attention.index_select(2, text_index)
            .index_select(3, image_index)
            .sum(dim=-1)
            .mean()
        )
        image_to_text = (
            attention.index_select(2, image_index)
            .index_select(3, text_index)
            .sum(dim=-1)
            .mean()
        )
        proxy = 0.5 * (text_to_image + image_to_text)
        if not torch.isfinite(proxy):
            raise ValueError("routing proxy is not finite")
        return float(proxy.item())

    def intervene(self, inputs, layer: int, group: str, mode: str):
        """Run one deterministic LM-encoder intervention and return a scalar logit.

        Callers MUST call ``get_target_logit(inputs, ...)`` on the same inputs
        before ``intervene`` so clean and intervened logits measure the same
        target token.

        ``layer`` indexes ``language_model.encoder.block`` from zero. Canonical
        groups are ``image``, ``text``, and ``negative_control``; canonical modes
        are ``ablate``, ``mask``, ``patch``, ``shuffle``, and ``noop``.
        ``mode='negative_control'`` remains a compatibility alias for the Lane B
        baseline and maps to padding-position ablation.

        The target token is the one previously resolved by
        :meth:`get_target_logit` on this input mapping. If no target metadata is
        present, the clean first-generated token is used. For patching, callers
        may attach prepared corrupt inputs under
        ``_crossroute_corrupt_inputs``; otherwise a deterministic blank-image or
        pad-token corruption is synthesized.
        """
        if mode == "negative_control":
            group = "negative_control"
            mode = "ablate"
        if mode not in _SUPPORTED_INTERVENTION_MODES:
            supported = ", ".join(sorted(_SUPPORTED_INTERVENTION_MODES))
            raise ValueError(f"unsupported intervention mode {mode!r}; expected {supported}")
        if group not in _SUPPORTED_INTERVENTION_GROUPS:
            supported = ", ".join(sorted(_SUPPORTED_INTERVENTION_GROUPS))
            raise ValueError(f"unsupported intervention group {group!r}; expected {supported}")
        if group == "negative_control" and mode == "patch":
            raise ValueError("patch mode requires the image or text group")

        layer_module = self._lm_encoder_layer(layer)
        target_token_id = self._intervention_target_token_id(inputs)

        def run_target_logit(run_inputs) -> float:
            _, first_step_logits = self._run_first_decode_step(run_inputs, capture=False)
            return self._target_logit_from_id(first_step_logits, target_token_id)

        if mode == "noop":
            hook = ActivationHook("noop")
            with managed_forward_hook(layer_module, hook):
                return run_target_logit(inputs)

        if group == "negative_control":
            negative_inputs, positions = self._with_padding_negative_control(inputs)
            behavior = {
                "ablate": "zero",
                "mask": "mean",
                "shuffle": "shuffle",
            }[mode]
            hook = ActivationHook(behavior, positions=positions)
            with managed_forward_hook(layer_module, hook):
                return run_target_logit(negative_inputs)

        if mode == "patch":
            corrupt_inputs = inputs.get(_CORRUPT_INPUTS_KEY)
            if corrupt_inputs is None:
                corrupt_inputs = self._make_corrupt_inputs(inputs, group)
            if not isinstance(corrupt_inputs, dict):
                raise TypeError(f"{_CORRUPT_INPUTS_KEY} must contain prepared input tensors")
            return float(
                run_activation_patching(
                    self,
                    inputs,
                    corrupt_inputs,
                    layer_module,
                    group,
                    run_target_logit,
                )
            )

        if mode in {"ablate", "mask"}:
            method = "zero" if mode == "ablate" else "mean"
            return float(
                run_ablation(
                    self,
                    inputs,
                    layer_module,
                    group,
                    method,
                    run_target_logit,
                )
            )

        hook = ActivationHook("shuffle", positions=group_positions(self, inputs, group))
        with managed_forward_hook(layer_module, hook):
            return run_target_logit(inputs)

    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        """Return one scalar logit from the first decoder step.

        Policy rules:

        - ``exact_token`` requires ``target_answer`` to tokenize to exactly one
          non-special token and returns that token's logit.
        - ``first_generated_token`` ignores ``target_answer``, greedily selects
          the model's highest-logit token, and returns its logit.
        - ``selected_candidate_logit`` tokenizes ``target_answer`` and returns
          the logit of its first non-special token. This permits multi-token
          candidate answers while keeping the audit unit at one decode step.
        """
        if policy not in _SUPPORTED_TARGET_POLICIES:
            supported = ", ".join(sorted(_SUPPORTED_TARGET_POLICIES))
            raise ValueError(f"unsupported target-token policy {policy!r}; expected one of {supported}")

        _, first_step_logits = self._run_first_decode_step(inputs, capture=False)
        if policy == "first_generated_token":
            token_id = int(first_step_logits[0].argmax().item())
        else:
            token_ids = self.processor.tokenizer.encode(
                target_answer,
                add_special_tokens=False,
            )
            if not token_ids:
                raise ValueError("target_answer did not produce any non-special tokens")
            if policy == "exact_token" and len(token_ids) != 1:
                raise ValueError(
                    "exact_token requires target_answer to tokenize to exactly one token; "
                    f"got {len(token_ids)}"
                )
            token_id = int(token_ids[0])

        value = self._target_logit_from_id(first_step_logits, token_id)
        inputs[_TARGET_TOKEN_ID_KEY] = token_id
        return value

    def run_controls(self, inputs, sample: dict) -> dict:
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for BLIP-2, but torch.cuda.is_available() is false")

        self._seed_everything()
        processor = Blip2Processor.from_pretrained(self.model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model = model.to(self.device)
        model.eval()

        processor.num_query_tokens = int(model.config.num_query_tokens)
        self.processor = processor
        self.model = model

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _read_seed() -> int:
        if not _DEFAULT_CONFIG_PATH.exists():
            return 1234
        with _DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        return int(config.get("seed", 1234))

    @staticmethod
    def _load_image(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.copy().convert("RGB")
        if isinstance(image, (str, Path)):
            image_path = Path(image).expanduser()
            if not image_path.is_file():
                raise FileNotFoundError(f"image does not exist: {image_path}")
            with Image.open(image_path) as opened:
                return opened.convert("RGB").copy()
        raise TypeError("image must be a filesystem path or PIL.Image.Image")

    def _run_first_decode_step(self, inputs, capture: bool):
        self._ensure_loaded()
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"pixel_values", "input_ids", "attention_mask", "inputs_embeds"}
        }
        if not model_inputs:
            raise ValueError("inputs do not contain BLIP-2 model tensors")

        if not self.model.config.use_decoder_only_language_model:
            batch_size = self._batch_size(model_inputs)
            start_token_id = self.model.config.text_config.decoder_start_token_id
            if start_token_id is None:
                start_token_id = self.model.config.text_config.pad_token_id
            model_inputs["decoder_input_ids"] = torch.full(
                (batch_size, 1),
                int(start_token_id),
                dtype=torch.long,
                device=self.device,
            )

        with torch.inference_mode(), self._capture_settings(capture):
            outputs = self.model(
                **model_inputs,
                output_hidden_states=capture,
                output_attentions=capture,
                return_dict=True,
                use_cache=False,
            )

        if self.model.config.use_decoder_only_language_model:
            first_step_logits = outputs.logits[:, -1, :]
        else:
            first_step_logits = outputs.logits[:, 0, :]
        return outputs, first_step_logits

    def _lm_encoder_layers(self):
        self._ensure_loaded()
        language_model = getattr(self.model, "language_model", None)
        encoder = getattr(language_model, "encoder", None)
        layers = getattr(encoder, "block", None)
        if layers is None:
            raise RuntimeError(
                "BLIP-2 language model does not expose encoder.block; "
                "Phase 3 interventions require an encoder-decoder checkpoint such as Flan-T5"
            )
        return layers

    def _lm_encoder_layer(self, layer: int):
        if not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        layers = self._lm_encoder_layers()
        if layer < 0 or layer >= len(layers):
            raise IndexError(
                f"LM encoder layer {layer} is outside valid range [0, {len(layers) - 1}]"
            )
        return layers[layer]

    def _intervention_target_token_id(self, inputs) -> int:
        stored = inputs.get(_TARGET_TOKEN_ID_KEY)
        if stored is not None:
            return int(stored)
        _, first_step_logits = self._run_first_decode_step(inputs, capture=False)
        return int(first_step_logits[0].argmax().item())

    @staticmethod
    def _target_logit_from_id(first_step_logits, token_id: int) -> float:
        if token_id < 0 or token_id >= first_step_logits.shape[-1]:
            raise ValueError(
                f"target token id {token_id} is outside vocabulary size "
                f"{first_step_logits.shape[-1]}"
            )
        value = first_step_logits[0, token_id]
        if not torch.isfinite(value):
            raise ValueError("target logit is not finite")
        return float(value.item())

    @staticmethod
    def _position_tensor(positions, sequence_length: int, device):
        if not positions:
            raise ValueError("routing proxy requires non-empty image and text groups")
        index = torch.tensor(positions, dtype=torch.long, device=device)
        if int(index.max().item()) >= sequence_length:
            raise IndexError(
                f"token position {int(index.max().item())} is outside "
                f"attention sequence length {sequence_length}"
            )
        return index

    @staticmethod
    def _clone_inputs(inputs) -> dict:
        cloned = {}
        for key, value in inputs.items():
            if key == _CORRUPT_INPUTS_KEY:
                continue
            cloned[key] = value.clone() if torch.is_tensor(value) else value
        return cloned

    def _make_corrupt_inputs(self, inputs, group: str) -> dict:
        corrupt = self._clone_inputs(inputs)
        if group == "image":
            pixel_values = corrupt.get("pixel_values")
            if pixel_values is None:
                raise ValueError("image patching requires pixel_values")
            corrupt["pixel_values"] = torch.zeros_like(pixel_values)
            return corrupt

        input_ids = corrupt.get("input_ids")
        if input_ids is None:
            raise ValueError("text patching requires input_ids")
        pad_token_id = self.model.config.text_config.pad_token_id
        if pad_token_id is None:
            raise ValueError("text patching requires a language-model pad token id")
        text_positions = group_positions(self, corrupt, "text")
        text_index = torch.tensor(text_positions, dtype=torch.long, device=input_ids.device)
        corrupt["input_ids"][0, text_index] = int(pad_token_id)
        return corrupt

    def _with_padding_negative_control(self, inputs) -> tuple[dict, tuple[int, ...]]:
        padded = self._clone_inputs(inputs)
        input_ids = padded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("negative control requires batched input_ids")
        attention_mask = padded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        pad_token_id = self.model.config.text_config.pad_token_id
        if pad_token_id is None:
            raise ValueError("negative control requires a language-model pad token id")
        pad_ids = torch.full(
            (input_ids.shape[0], 1),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        pad_mask = torch.zeros(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        padded["input_ids"] = torch.cat([input_ids, pad_ids], dim=1)
        padded["attention_mask"] = torch.cat([attention_mask, pad_mask], dim=1)
        return padded, (int(padded["input_ids"].shape[1] - 1),)

    @staticmethod
    def _batch_size(inputs) -> int:
        for key in ("input_ids", "pixel_values", "inputs_embeds"):
            value = inputs.get(key)
            if torch.is_tensor(value):
                return int(value.shape[0])
        raise ValueError("could not infer batch size from inputs")

    @contextmanager
    def _capture_settings(self, capture: bool) -> Iterator[None]:
        configs = [
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "vision_model", None), "config", None),
            getattr(getattr(self.model, "qformer", None), "config", None),
            getattr(getattr(self.model, "language_model", None), "config", None),
        ]
        previous = []
        for config in configs:
            if config is None:
                continue
            state = {
                "config": config,
                "output_hidden_states": getattr(config, "output_hidden_states", None),
                "output_attentions": getattr(config, "output_attentions", None),
            }
            previous.append(state)
            config.output_hidden_states = capture
            config.output_attentions = capture
        try:
            yield
        finally:
            for state in previous:
                config = state["config"]
                config.output_hidden_states = state["output_hidden_states"]
                config.output_attentions = state["output_attentions"]

    @classmethod
    def _summarize_hidden_states(cls, outputs) -> dict[str, Any]:
        language_outputs = getattr(outputs, "language_model_outputs", SimpleNamespace())
        return {
            "vision": cls._summarize_collection(
                getattr(getattr(outputs, "vision_outputs", None), "hidden_states", None)
            ),
            "qformer": cls._summarize_collection(
                getattr(getattr(outputs, "qformer_outputs", None), "hidden_states", None)
            ),
            "language_encoder": cls._summarize_collection(
                getattr(language_outputs, "encoder_hidden_states", None)
            ),
            "language_decoder": cls._summarize_collection(
                getattr(language_outputs, "decoder_hidden_states", None)
                or getattr(language_outputs, "hidden_states", None)
            ),
        }

    @classmethod
    def _summarize_attentions(cls, outputs) -> dict[str, Any]:
        qformer_outputs = getattr(outputs, "qformer_outputs", None)
        language_outputs = getattr(outputs, "language_model_outputs", SimpleNamespace())
        return {
            "vision": cls._summarize_collection(
                getattr(getattr(outputs, "vision_outputs", None), "attentions", None)
            ),
            "qformer_self": cls._summarize_collection(
                getattr(qformer_outputs, "attentions", None)
            ),
            "qformer_cross": cls._summarize_collection(
                getattr(qformer_outputs, "cross_attentions", None)
            ),
            "language_encoder": cls._summarize_collection(
                getattr(language_outputs, "encoder_attentions", None)
            ),
            "language_decoder": cls._summarize_collection(
                getattr(language_outputs, "decoder_attentions", None)
                or getattr(language_outputs, "attentions", None)
            ),
            "language_cross": cls._summarize_collection(
                getattr(language_outputs, "cross_attentions", None)
            ),
        }

    @staticmethod
    def _summarize_collection(collection) -> list[dict[str, Any]]:
        if collection is None:
            return []
        summaries = []
        for tensor in collection:
            if not torch.is_tensor(tensor):
                continue
            detached = tensor.detach()
            summaries.append(
                {
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype),
                    "mean": float(detached.float().mean().item()),
                }
            )
        return summaries
