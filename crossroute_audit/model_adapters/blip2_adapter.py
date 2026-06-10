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

from .base import ForwardOutput, ModelAdapter, TokenGroups


LOGGER = logging.getLogger(__name__)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
_SUPPORTED_TARGET_POLICIES = {
    "exact_token",
    "first_generated_token",
    "selected_candidate_logit",
}


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

        ``image`` may be a filesystem path or a ``PIL.Image.Image``. Images are
        converted to RGB and copied before processing so caller-owned image
        objects are not mutated. Integer token tensors stay integer while
        floating image tensors are moved to the model device and dtype.
        """
        self._ensure_loaded()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        pil_image = self._load_image(image)
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

        BLIP-2 prepends ``num_query_tokens`` projected image-query embeddings to
        the processor's text tokens. ``image`` contains the prepended indices,
        ``text`` contains every non-padding prompt index shifted by that image
        prefix, ``fusion`` covers the combined image-plus-text sequence, and
        ``answer`` is decoder position zero (the audited first token).
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
            valid_length = int(attention_mask[0].sum().item())
        else:
            valid_length = int(input_ids.shape[1])
        total_length = num_query_tokens + valid_length

        groups = TokenGroups(
            image=list(range(num_query_tokens)),
            text=list(range(num_query_tokens, total_length)),
            fusion=list(range(total_length)),
            answer=[0],
        )
        LOGGER.debug(
            "BLIP-2 token groups: processor_input_shape=%s combined_length=%d "
            "image=%d text=%d fusion=%d answer=%d",
            list(input_ids.shape),
            total_length,
            len(groups.image),
            len(groups.text),
            len(groups.fusion),
            len(groups.answer),
        )
        return groups

    def get_layer_count(self) -> int:
        """Return the number of Q-Former encoder layers, not LM decoder layers."""
        self._ensure_loaded()
        return int(self.model.config.qformer_config.num_hidden_layers)

    def get_routing_proxy(self, inputs, layer: int):
        raise NotImplementedError

    def intervene(self, inputs, layer: int, group: str, mode: str):
        raise NotImplementedError

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

        if token_id < 0 or token_id >= first_step_logits.shape[-1]:
            raise ValueError(
                f"target token id {token_id} is outside vocabulary size "
                f"{first_step_logits.shape[-1]}"
            )
        value = first_step_logits[0, token_id]
        if not torch.isfinite(value):
            raise ValueError("target logit is not finite")
        return float(value.item())

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
