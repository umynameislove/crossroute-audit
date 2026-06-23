"""Qwen2-VL decoder-only adapter.

This module owns the Qwen2-VL-specific PyTorch and Transformers details. The
shared audit pipeline continues to interact only with the architecture-agnostic
``ModelAdapter`` contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Iterator

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .llava_adapter import LLaVAAdapter


LOGGER = logging.getLogger(__name__)
_QWEN_MODEL_INPUT_KEYS = {
    "pixel_values",
    "pixel_values_videos",
    "input_ids",
    "attention_mask",
    "inputs_embeds",
    "image_grid_thw",
    "video_grid_thw",
    "position_ids",
    "rope_deltas",
    "cache_position",
}
_QWEN_ATTRIBUTION_POSITION_IDS_KEY = "_crossroute_qwenvl_position_ids"
_IMAGE_PAD_TOKEN = "<|image_pad|>"


class QwenVLAdapter(LLaVAAdapter):
    """Adapter for ``Qwen/Qwen2-VL-7B-Instruct``.

    Qwen2-VL is decoder-only for CrossRoute's audit purposes. The audit layer
    axis is the ordered LLM decoder stack at ``model.model.language_model.layers``.
    Qwen image-token spans are dynamic, so image positions are detected from
    every ``input_ids == image_token_id`` location rather than a fixed sequence
    length.
    """

    name = "qwenvl"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
    ):
        super().__init__(model_name=model_name, device=device)
        self.processor: AutoProcessor | None = None
        self.model: Qwen2VLForConditionalGeneration | None = None

    def prepare_inputs(self, image, question: str) -> dict:
        """Prepare one deterministic Qwen2-VL chat prompt and image.

        ``image`` may be a filesystem path, a ``PIL.Image.Image``, or ``None``
        for a no-image baseline. Qwen2-VL requires ``image_grid_thw`` alongside
        ``pixel_values``; both are preserved in the returned mapping. For the
        no-image baseline, a black image with the normal chat structure is used
        and ``pixel_values`` is zeroed after processor tokenization.
        """
        self._ensure_loaded()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        zero_pixels = image is None
        image_size = int(getattr(self.model.config.vision_config, "image_size", 336))
        pil_image = (
            Image.new("RGB", (image_size, image_size), 0)
            if zero_pixels
            else self._load_image(image)
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question.strip()},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.processor(
            images=[pil_image],
            text=[prompt],
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
        self._validate_qwen_inputs(prepared)

        LOGGER.debug(
            "Prepared Qwen2-VL inputs: %s",
            {key: list(value.shape) for key, value in prepared.items() if torch.is_tensor(value)},
        )
        return prepared

    def layer_axis_name(self) -> str:
        """Return the Qwen2-VL decoder-layer module path used as the audit axis."""
        return "model.model.language_model.layers"

    def forward_target_logit_from_embeddings(
        self,
        embeddings,
        attention_mask,
        target_token_id: int,
    ):
        """Return differentiable Qwen2-VL next-token logits from decoder embeddings.

        ``prepare_attribution_inputs`` must be called first so the adapter can
        reuse the exact multi-modal position ids that Qwen2-VL computed for the
        clean prompt. The returned tensor is ``logits[:, -1, target_token_id]``:
        the target score for the next token after the prompt.
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

        position_ids = getattr(self, _QWEN_ATTRIBUTION_POSITION_IDS_KEY, None)
        if position_ids is None:
            raise RuntimeError(
                "Qwen2-VL attribution forward requires prepare_attribution_inputs "
                "to capture multi-modal position ids first"
            )
        position_ids = self._expanded_position_ids(
            position_ids,
            embeddings.shape[0],
            embeddings.shape[1],
            embeddings.device,
        )

        with self.attribution_float32(), torch.enable_grad():
            float_embeddings = embeddings.to(dtype=torch.float32)
            outputs = self._language_model()(
                inputs_embeds=float_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
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

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Qwen2-VL, but torch.cuda.is_available() is false")

        self._seed_everything()
        processor = AutoProcessor.from_pretrained(self.model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model = model.to(self.device)
        model.eval()
        self.processor = processor
        self.model = model

    def _run_first_decode_step(self, inputs, capture: bool):
        self._ensure_loaded()
        model_inputs = self._qwen_model_inputs(inputs)

        with torch.inference_mode(), self._capture_settings(capture):
            outputs = self.model(
                **model_inputs,
                output_hidden_states=capture,
                output_attentions=capture,
                return_dict=True,
                use_cache=False,
            )

        first_step_logits = outputs.logits[:, -1, :]
        return outputs, first_step_logits

    def _capture_decoder_embeddings(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("attribution inputs must contain batched input_ids")

        model_inputs = self._qwen_model_inputs(inputs)
        captured_embeddings = []
        captured_attention_masks = []
        captured_position_ids = []

        def capture_decoder_inputs(module, args, kwargs):
            del module
            decoder_inputs = kwargs.get("inputs_embeds")
            if decoder_inputs is None and len(args) >= 4:
                decoder_inputs = args[3]
            if not torch.is_tensor(decoder_inputs) or decoder_inputs.ndim != 3:
                raise RuntimeError(
                    "Qwen2-VL did not pass [batch, sequence, hidden] embeddings "
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

            position_ids = kwargs.get("position_ids")
            if position_ids is None and len(args) >= 3:
                position_ids = args[2]
            if not torch.is_tensor(position_ids) or position_ids.ndim not in {2, 3}:
                raise RuntimeError(
                    "Qwen2-VL did not pass 2D or 3D position ids to the decoder"
                )

            captured_embeddings.append(decoder_inputs.detach())
            captured_attention_masks.append(attention_mask.detach())
            captured_position_ids.append(position_ids.detach())

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
        position_ids = captured_position_ids[0].clone()
        if decoder_embeddings.shape[:2] != attention_mask.shape:
            raise ValueError(
                "decoder embeddings and attention_mask have inconsistent shapes"
            )
        self._cache_merged_positions(inputs, decoder_embeddings, attention_mask)
        setattr(self, _QWEN_ATTRIBUTION_POSITION_IDS_KEY, position_ids)
        return decoder_embeddings, attention_mask

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
        if int(input_ids.shape[1]) != int(merged_sequence_length):
            raise ValueError(
                "Qwen2-VL decoder sequence length must match input_ids because "
                "image features are scattered into dynamic image-token positions"
            )

        image_token_id = self._image_token_id()
        valid_positions = self._valid_positions(merged_attention_mask, merged_sequence_length)
        image_positions = tuple(
            position
            for position in valid_positions
            if int(input_ids[0, position].item()) == image_token_id
        )
        if not image_positions:
            raise ValueError("Qwen2-VL inputs do not contain image placeholder tokens")

        image_set = set(image_positions)
        text_positions = tuple(
            position for position in valid_positions if position not in image_set
        )
        return image_positions, text_positions, tuple(valid_positions)

    def _expected_image_token_count(self) -> int | None:
        return None

    def _image_token_id(self) -> int:
        token_id = getattr(self.model.config, "image_token_id", None)
        if token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None:
                token_id = tokenizer.convert_tokens_to_ids(_IMAGE_PAD_TOKEN)
        if token_id is None:
            raise ValueError("Qwen2-VL config/tokenizer does not define an image token id")
        return int(token_id)

    def _language_model(self):
        self._ensure_loaded()
        core_model = getattr(self.model, "model", None)
        language_model = getattr(core_model, "language_model", None)
        if language_model is None:
            language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Qwen2-VL model does not expose a decoder language_model")
        return language_model

    def _qwen_model_inputs(self, inputs: dict) -> dict:
        model_inputs = {
            key: value for key, value in inputs.items() if key in _QWEN_MODEL_INPUT_KEYS
        }
        if not model_inputs:
            raise ValueError("inputs do not contain Qwen2-VL model tensors")
        self._validate_qwen_inputs(model_inputs)
        return model_inputs

    @staticmethod
    def _validate_qwen_inputs(inputs: dict) -> None:
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("Qwen2-VL inputs require input_ids with shape [batch, sequence]")
        pixel_values = inputs.get("pixel_values")
        if not torch.is_tensor(pixel_values) or pixel_values.ndim != 2:
            raise ValueError("Qwen2-VL inputs require flattened 2D pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if not torch.is_tensor(image_grid_thw) or image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
            raise ValueError("Qwen2-VL inputs require image_grid_thw with shape [num_images, 3]")
        if not bool(torch.isfinite(pixel_values).all()):
            raise ValueError("Qwen2-VL pixel_values contain non-finite values")

    @staticmethod
    def _expanded_position_ids(
        position_ids: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        device,
    ) -> torch.Tensor:
        if not torch.is_tensor(position_ids) or position_ids.ndim not in {2, 3}:
            raise ValueError("Qwen2-VL position_ids must be 2D or 3D")
        ids = position_ids.to(device=device)
        if ids.ndim == 2:
            if ids.shape[-1] != sequence_length:
                raise ValueError("Qwen2-VL 2D position_ids sequence length mismatch")
            if ids.shape[0] == 1 and batch_size != 1:
                ids = ids.expand(batch_size, -1)
            if ids.shape[0] != batch_size:
                raise ValueError("Qwen2-VL 2D position_ids batch size mismatch")
            return ids

        if ids.shape[-1] != sequence_length:
            raise ValueError("Qwen2-VL 3D position_ids sequence length mismatch")
        if ids.shape[1] == 1 and batch_size != 1:
            ids = ids.expand(-1, batch_size, -1)
        if ids.shape[1] != batch_size:
            raise ValueError("Qwen2-VL 3D position_ids batch size mismatch")
        return ids

    @contextmanager
    def _capture_settings(self, capture: bool) -> Iterator[None]:
        core_model = getattr(self.model, "model", None)
        configs = [
            getattr(self.model, "config", None),
            getattr(core_model, "config", None),
            getattr(getattr(core_model, "visual", None), "config", None),
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
