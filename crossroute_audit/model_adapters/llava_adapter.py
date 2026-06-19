"""LLaVA-1.5 decoder-only adapter.

This module owns the LLaVA-specific PyTorch and Transformers details. Shared
audit code interacts only with the architecture-agnostic ``ModelAdapter``
contract.
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
from transformers import LlavaForConditionalGeneration, LlavaProcessor
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
_MERGED_IMAGE_POSITIONS_KEY = "_crossroute_merged_image_positions"
_MERGED_TEXT_POSITIONS_KEY = "_crossroute_merged_text_positions"
_MERGED_FUSION_POSITIONS_KEY = "_crossroute_merged_fusion_positions"
_MERGED_ATTENTION_MASK_KEY = "_crossroute_merged_attention_mask"
_PROMPT_TEMPLATE = "USER: <image>\n{question} ASSISTANT:"


class LLaVAAdapter(ModelAdapter):
    """Adapter for ``llava-hf/llava-1.5-7b-hf``.

    The checkpoint is loaded lazily. LLaVA-1.5 is decoder-only: the audit layer
    axis is the ordered LLM decoder-layer stack at ``model.language_model.layers``.
    """

    name = "llava"

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.seed = self._read_seed()
        self.processor: LlavaProcessor | None = None
        self.model: LlavaForConditionalGeneration | None = None

    def prepare_inputs(self, image, question: str) -> dict:
        """Prepare one deterministic LLaVA prompt and image.

        ``image`` may be a filesystem path, a ``PIL.Image.Image``, or ``None``
        for a no-image baseline. The prompt is formatted as
        ``USER: <image>\n{question} ASSISTANT:`` unless the caller already
        provides a formatted LLaVA prompt. Floating tensors are moved to the
        adapter device in fp16; integer tensors remain integer.
        """
        self._ensure_loaded()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        prompt = self._format_prompt(question.strip())
        zero_pixels = image is None
        image_size = int(getattr(self.model.config.vision_config, "image_size", 336))
        pil_image = (
            Image.new("RGB", (image_size, image_size), 0)
            if zero_pixels
            else self._load_image(image)
        )
        encoded = self.processor(
            images=pil_image,
            text=prompt,
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
            prepared["pixel_values"] = torch.zeros_like(prepared["pixel_values"])

        LOGGER.debug(
            "Prepared LLaVA inputs: %s",
            {key: list(value.shape) for key, value in prepared.items() if torch.is_tensor(value)},
        )
        return prepared

    def forward(self, inputs, capture: bool = False) -> ForwardOutput:
        """Run a clean decoder-only next-token forward pass."""
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
        """Return image/text positions in the merged decoder input sequence."""
        self._ensure_loaded()
        image_positions, text_positions, fusion_positions = self._merged_positions(inputs)
        groups = TokenGroups(
            image=list(image_positions),
            text=list(text_positions),
            fusion=list(fusion_positions),
            answer=[],
        )
        LOGGER.debug(
            "LLaVA token groups: image=%d text=%d fusion=%d answer=%d",
            len(groups.image),
            len(groups.text),
            len(groups.fusion),
            len(groups.answer),
        )
        return groups

    def get_layer_count(self) -> int:
        """Return the decoder-layer count for this decoder-only adapter."""
        return self.get_intervention_layer_count()

    def get_intervention_layer_count(self) -> int:
        """Return the number of LLM decoder layers on the audit layer axis."""
        self._ensure_loaded()
        return len(self._decoder_layers())

    def layer_axis_name(self) -> str:
        """Return the LLaVA decoder-layer module path used as the audit axis."""
        return "model.language_model.layers"

    def prepare_attribution_inputs(
        self,
        inputs,
        target_answer: str,
        policy: str,
    ):
        """Capture differentiable merged decoder embeddings for Layer-IG.

        The returned embeddings are the exact tensor passed to LLaVA's decoder
        after image placeholders have been replaced by projected CLIP features.
        The target token is resolved before capture so attribution, causal
        intervention, and clean forward all measure the same next-token logit.
        """
        self._ensure_loaded()
        self.get_target_logit(inputs, target_answer, policy)
        target_token_id = int(inputs[_TARGET_TOKEN_ID_KEY])

        decoder_embeddings, attention_mask = self._capture_decoder_embeddings(inputs)
        return (
            decoder_embeddings.detach().requires_grad_(True),
            attention_mask.detach(),
            target_token_id,
        )

    def attribution_baseline_embeddings(self, inputs):
        """Return merged decoder embeddings for the same prompt with a black image."""
        self._ensure_loaded()
        pixel_values = inputs.get("pixel_values")
        if not torch.is_tensor(pixel_values):
            raise ValueError("attribution baseline requires pixel_values")
        blank_inputs = self._clone_inputs(inputs)
        blank_inputs["pixel_values"] = torch.zeros_like(pixel_values)
        self._clear_merged_metadata(blank_inputs)
        baseline_embeddings, _ = self._capture_decoder_embeddings(blank_inputs)
        return baseline_embeddings.detach()

    def forward_target_logit_from_embeddings(
        self,
        embeddings,
        attention_mask,
        target_token_id: int,
    ):
        """Return differentiable next-token logits from merged decoder embeddings.

        ``embeddings`` must have shape ``[batch, sequence, hidden]`` and
        ``attention_mask`` must have shape ``[batch, sequence]``. The returned
        tensor is ``logits[:, -1, target_token_id]``: the score of the target as
        the next token after the prompt. No decoder-start token is used for
        decoder-only LLaVA.
        """
        self._ensure_loaded()
        if not torch.is_tensor(embeddings) or embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if embeddings.shape[:2] != attention_mask.shape:
            raise ValueError(
                "embeddings and attention_mask must share batch and sequence dimensions"
            )
        if not isinstance(target_token_id, int):
            raise TypeError("target_token_id must be an integer")

        vocab_size = int(self.model.config.text_config.vocab_size)
        if target_token_id < 0 or target_token_id >= vocab_size:
            raise ValueError(
                f"target token id {target_token_id} is outside vocabulary size {vocab_size}"
            )

        with self.attribution_float32(), torch.enable_grad():
            float_embeddings = embeddings.to(dtype=torch.float32)
            outputs = self._language_model()(
                inputs_embeds=float_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
            logits = self.model.lm_head(hidden_states)
            target_logits = logits[:, -1, target_token_id].to(dtype=torch.float32)

        if not bool(torch.isfinite(target_logits).all()):
            raise ValueError("differentiable target logit is not finite")
        return target_logits

    @contextmanager
    def attribution_float32(self):
        """Temporarily cast the decoder and LM head to float32 for attribution."""
        self._ensure_loaded()
        modules = (self._language_model(), self.model.lm_head)
        depth = int(getattr(self, "_attribution_float32_depth", 0))
        if depth > 0:
            self._attribution_float32_depth = depth + 1
            try:
                yield
            finally:
                self._attribution_float32_depth -= 1
            return

        parameters = self._unique_float_parameters(modules)
        parameter_dtypes = {parameter.dtype for parameter in parameters}
        if not parameter_dtypes:
            raise RuntimeError("LLaVA attribution path has no floating-point parameters")
        if len(parameter_dtypes) != 1:
            dtypes = ", ".join(sorted(str(dtype) for dtype in parameter_dtypes))
            raise RuntimeError(
                "attribution requires one decoder parameter dtype; "
                f"found {dtypes}"
            )

        original_dtype = next(iter(parameter_dtypes))
        original_requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        converted = original_dtype != torch.float32
        try:
            if converted:
                for module in modules:
                    module.to(dtype=torch.float32)
        except Exception:
            if converted:
                for module in modules:
                    module.to(dtype=original_dtype)
            raise

        for parameter in parameters:
            parameter.requires_grad_(False)

        self._attribution_float32_depth = 1
        try:
            yield
        finally:
            self._attribution_float32_depth = 0
            for parameter, requires_grad in zip(
                parameters,
                original_requires_grad,
                strict=True,
            ):
                parameter.requires_grad_(requires_grad)
            if converted:
                for module in modules:
                    module.to(dtype=original_dtype)

    @contextmanager
    def attribution_layer_output(self, layer: int):
        """Expose one decoder layer's primary hidden-state output to Captum."""
        layer_module = self._decoder_layer(layer)
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
                "decoder layer output must be a tensor or a sequence whose "
                "first item is a tensor"
            )

        handle = layer_module.register_forward_hook(route_hidden_state)
        try:
            yield tap
        finally:
            handle.remove()

    def get_routing_proxy(self, inputs, layer: int):
        """Return cross-modal self-attention mass at one decoder layer."""
        self._decoder_layer(layer)
        outputs, _ = self._run_first_decode_step(inputs, capture=True)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None or layer >= len(attentions) or attentions[layer] is None:
            raise RuntimeError(
                "decoder attentions were not materialized; "
                "load LLaVA with attn_implementation='eager'"
            )

        attention = attentions[layer]
        if attention.ndim != 4:
            raise ValueError(
                "decoder attention must have shape [batch, heads, query, key], "
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
        """Run one decoder-layer intervention and return a scalar target logit.

        Callers MUST call ``get_target_logit(inputs, ...)`` on the same inputs
        before ``intervene`` so clean and intervened logits measure the same
        target token. ``layer`` indexes ``model.language_model.layers`` from
        zero. ``mode='negative_control'`` remains a compatibility alias for
        padding-position ablation.
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

        layer_module = self._decoder_layer(layer)
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
        """Return one next-token scalar logit under the target-token policy."""
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
        """Run the standard control suite through the shared controls module."""
        del inputs
        from crossroute_audit.controls.baselines import run_controls

        return run_controls(self, sample)

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for LLaVA, but torch.cuda.is_available() is false")

        self._seed_everything()
        processor = LlavaProcessor.from_pretrained(self.model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model = model.to(self.device)
        model.eval()
        self._configure_processor(processor, model)
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

    @staticmethod
    def _format_prompt(question: str) -> str:
        if "<image>" in question and "ASSISTANT:" in question:
            return question
        return _PROMPT_TEMPLATE.format(question=question)

    @staticmethod
    def _configure_processor(processor, model) -> None:
        vision_config = model.config.vision_config
        processor.patch_size = int(getattr(vision_config, "patch_size", 14))
        processor.vision_feature_select_strategy = getattr(
            model.config,
            "vision_feature_select_strategy",
            "default",
        )
        processor.num_additional_image_tokens = 1

    def _run_first_decode_step(self, inputs, capture: bool):
        self._ensure_loaded()
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {
                "pixel_values",
                "input_ids",
                "attention_mask",
                "inputs_embeds",
                "image_sizes",
                "position_ids",
            }
        }
        if not model_inputs:
            raise ValueError("inputs do not contain LLaVA model tensors")

        with torch.inference_mode(), self._capture_settings(capture):
            outputs = self.model(
                **model_inputs,
                output_hidden_states=capture,
                output_attentions=capture,
                return_dict=True,
                use_cache=False,
                logits_to_keep=1,
            )

        first_step_logits = outputs.logits[:, -1, :]
        return outputs, first_step_logits

    def _capture_decoder_embeddings(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("attribution inputs must contain batched input_ids")

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {
                "pixel_values",
                "input_ids",
                "attention_mask",
                "inputs_embeds",
                "image_sizes",
                "position_ids",
            }
        }
        captured_embeddings = []
        captured_attention_masks = []

        def capture_decoder_inputs(module, args, kwargs):
            del module
            decoder_inputs = kwargs.get("inputs_embeds")
            if decoder_inputs is None and len(args) >= 4:
                decoder_inputs = args[3]
            if not torch.is_tensor(decoder_inputs) or decoder_inputs.ndim != 3:
                raise RuntimeError(
                    "LLaVA did not pass [batch, sequence, hidden] embeddings "
                    "to the decoder"
                )
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(args) >= 2:
                attention_mask = args[1]
            if attention_mask is None:
                attention_mask = torch.ones(
                    decoder_inputs.shape[:2],
                    dtype=torch.long,
                    device=decoder_inputs.device,
                )
            if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
                raise RuntimeError("decoder attention_mask must have shape [batch, sequence]")
            captured_embeddings.append(decoder_inputs.detach())
            captured_attention_masks.append(attention_mask.detach())

        language_model = self._language_model()
        handle = language_model.register_forward_pre_hook(
            capture_decoder_inputs,
            with_kwargs=True,
        )
        try:
            with torch.no_grad(), self._capture_settings(False):
                self.model.model(
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
                "expected exactly one decoder call while preparing attribution "
                f"inputs, got {len(captured_embeddings)}"
            )
        decoder_embeddings = captured_embeddings[0].clone()
        attention_mask = captured_attention_masks[0].clone()
        if decoder_embeddings.shape[:2] != attention_mask.shape:
            raise ValueError(
                "decoder embeddings and attention_mask have inconsistent shapes"
            )
        self._cache_merged_positions(inputs, decoder_embeddings, attention_mask)
        return decoder_embeddings, attention_mask

    def _cache_merged_positions(
        self,
        inputs: dict,
        decoder_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        image_positions, text_positions, fusion_positions = self._positions_from_input_ids(
            inputs,
            merged_sequence_length=int(decoder_embeddings.shape[1]),
            merged_attention_mask=attention_mask,
        )
        inputs[_MERGED_IMAGE_POSITIONS_KEY] = tuple(image_positions)
        inputs[_MERGED_TEXT_POSITIONS_KEY] = tuple(text_positions)
        inputs[_MERGED_FUSION_POSITIONS_KEY] = tuple(fusion_positions)
        inputs[_MERGED_ATTENTION_MASK_KEY] = attention_mask.detach()

    def _merged_positions(self, inputs) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        cached = (
            inputs.get(_MERGED_IMAGE_POSITIONS_KEY),
            inputs.get(_MERGED_TEXT_POSITIONS_KEY),
            inputs.get(_MERGED_FUSION_POSITIONS_KEY),
        )
        if all(value is not None for value in cached):
            return tuple(cached[0]), tuple(cached[1]), tuple(cached[2])
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("get_token_groups requires batched input_ids")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self._positions_from_input_ids(
            inputs,
            merged_sequence_length=int(input_ids.shape[1]),
            merged_attention_mask=attention_mask,
        )

    def _positions_from_input_ids(
        self,
        inputs: dict,
        *,
        merged_sequence_length: int,
        merged_attention_mask: torch.Tensor,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("token grouping currently supports one sample at a time")
        if not torch.is_tensor(merged_attention_mask) or merged_attention_mask.ndim != 2:
            raise ValueError("merged attention_mask must have shape [batch, sequence]")
        if merged_attention_mask.shape[0] != 1:
            raise ValueError("token grouping currently supports one sample at a time")

        image_token_id = self._image_token_id()
        valid_positions = self._valid_positions(merged_attention_mask, merged_sequence_length)
        original_valid = self._valid_positions(
            inputs.get("attention_mask", torch.ones_like(input_ids)),
            int(input_ids.shape[1]),
        )
        placeholder_positions = tuple(
            position
            for position in original_valid
            if int(input_ids[0, position].item()) == image_token_id
        )
        if not placeholder_positions:
            raise ValueError("LLaVA inputs do not contain image placeholder tokens")

        if int(input_ids.shape[1]) == merged_sequence_length:
            image_positions = tuple(
                position for position in valid_positions if position in placeholder_positions
            )
            expected = self._expected_image_token_count()
            if expected is not None and len(image_positions) != expected:
                raise ValueError(
                    "processor/model image token count mismatch: "
                    f"expected {expected}, got {len(image_positions)}"
                )
            image_set = set(image_positions)
            text_positions = tuple(
                position for position in valid_positions if position not in image_set
            )
            return image_positions, text_positions, tuple(valid_positions)

        if len(placeholder_positions) != 1:
            raise ValueError(
                "merged decoder sequence length differs from input_ids, but "
                f"found {len(placeholder_positions)} image placeholders"
            )
        placeholder = placeholder_positions[0]
        image_count = merged_sequence_length - (len(original_valid) - 1)
        if image_count <= 0:
            raise ValueError("could not infer merged image-token span")
        image_positions = tuple(range(placeholder, placeholder + image_count))
        shifted_text = []
        for position in original_valid:
            if position == placeholder:
                continue
            if position < placeholder:
                shifted_text.append(position)
            else:
                shifted_text.append(position + image_count - 1)
        valid_set = set(valid_positions)
        text_positions = tuple(position for position in shifted_text if position in valid_set)
        return image_positions, text_positions, tuple(valid_positions)

    def _expected_image_token_count(self) -> int | None:
        value = getattr(self.model.config, "image_seq_length", None)
        if value is None:
            return None
        return int(value)

    def _image_token_id(self) -> int:
        token_id = getattr(self.model.config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(self.model.config, "image_token_index", None)
        if token_id is None:
            raise ValueError("LLaVA config does not define an image token id")
        return int(token_id)

    @staticmethod
    def _valid_positions(attention_mask: torch.Tensor, sequence_length: int) -> tuple[int, ...]:
        if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.shape[1] > sequence_length:
            raise ValueError("attention_mask is longer than the merged sequence")
        positions = attention_mask[0].nonzero(as_tuple=False).flatten().tolist()
        return tuple(int(position) for position in positions if int(position) < sequence_length)

    @staticmethod
    def _clear_merged_metadata(inputs: dict) -> None:
        for key in (
            _MERGED_IMAGE_POSITIONS_KEY,
            _MERGED_TEXT_POSITIONS_KEY,
            _MERGED_FUSION_POSITIONS_KEY,
            _MERGED_ATTENTION_MASK_KEY,
        ):
            inputs.pop(key, None)

    def _language_model(self):
        self._ensure_loaded()
        core_model = getattr(self.model, "model", None)
        language_model = getattr(core_model, "language_model", None)
        if language_model is None:
            language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            raise RuntimeError("LLaVA model does not expose a decoder language_model")
        return language_model

    def _decoder_layers(self):
        language_model = self._language_model()
        layers = getattr(language_model, "layers", None)
        if layers is None:
            nested_model = getattr(language_model, "model", None)
            layers = getattr(nested_model, "layers", None)
        if layers is None:
            raise RuntimeError("LLaVA language model does not expose decoder layers")
        return layers

    def _decoder_layer(self, layer: int):
        if not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        layers = self._decoder_layers()
        if layer < 0 or layer >= len(layers):
            raise IndexError(
                f"decoder layer {layer} is outside valid range [0, {len(layers) - 1}]"
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
        self._clear_merged_metadata(corrupt)
        if group == "image":
            pixel_values = corrupt.get("pixel_values")
            if pixel_values is None:
                raise ValueError("image patching requires pixel_values")
            corrupt["pixel_values"] = torch.zeros_like(pixel_values)
            return corrupt

        input_ids = corrupt.get("input_ids")
        if input_ids is None:
            raise ValueError("text patching requires input_ids")
        pad_token_id = self._pad_token_id()
        text_positions = group_positions(self, corrupt, "text")
        text_index = torch.tensor(text_positions, dtype=torch.long, device=input_ids.device)
        corrupt["input_ids"][0, text_index] = int(pad_token_id)
        return corrupt

    def _with_padding_negative_control(self, inputs) -> tuple[dict, tuple[int, ...]]:
        padded = self._clone_inputs(inputs)
        self._clear_merged_metadata(padded)
        input_ids = padded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("negative control requires batched input_ids")
        attention_mask = padded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        pad_token_id = self._pad_token_id()
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

    def _pad_token_id(self) -> int:
        candidates = [
            getattr(self.model.config.text_config, "pad_token_id", None),
            getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None),
            getattr(self.model.config.text_config, "eos_token_id", None),
        ]
        for candidate in candidates:
            if candidate is not None:
                return int(candidate)
        raise ValueError("LLaVA text corruption requires a pad or eos token id")

    @staticmethod
    def _unique_float_parameters(modules) -> tuple[torch.nn.Parameter, ...]:
        parameters = []
        seen = set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) in seen or not parameter.is_floating_point():
                    continue
                seen.add(id(parameter))
                parameters.append(parameter)
        return tuple(parameters)

    @contextmanager
    def _capture_settings(self, capture: bool) -> Iterator[None]:
        core_model = getattr(self.model, "model", None)
        configs = [
            getattr(self.model, "config", None),
            getattr(core_model, "config", None),
            getattr(getattr(core_model, "vision_tower", None), "config", None),
            getattr(self._language_model(), "config", None),
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
        return {
            "decoder": cls._summarize_collection(getattr(outputs, "hidden_states", None)),
            "image": cls._summarize_collection(getattr(outputs, "image_hidden_states", None)),
        }

    @classmethod
    def _summarize_attentions(cls, outputs) -> dict[str, Any]:
        return {
            "decoder": cls._summarize_collection(getattr(outputs, "attentions", None)),
        }

    @staticmethod
    def _summarize_collection(collection) -> list[dict[str, Any]]:
        if collection is None:
            return []
        if torch.is_tensor(collection):
            iterable = (collection,)
        else:
            iterable = collection
        summaries = []
        for tensor in iterable:
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
