"""InstructBLIP adapter for the encoder-decoder audit path.

InstructBLIP-FlanT5 shares the same T5 language-model intervention surface as
BLIP-2, but its Q-Former is instruction-aware. This adapter reuses the
validated BLIP-2 audit mechanics and overrides only the model-specific input
preparation and encoder-embedding capture path.
"""
from __future__ import annotations

import logging

from PIL import Image
import torch
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from .blip2_adapter import BLIP2Adapter


LOGGER = logging.getLogger(__name__)


class InstructBLIPAdapter(BLIP2Adapter):
    """Adapter for ``Salesforce/instructblip-flan-t5-xl``.

    The audit layer axis is the T5 encoder stack at
    ``language_model.encoder.block``. The adapter lets the model perform its own
    instruction-aware Q-Former computation, then captures the exact embeddings
    passed into the T5 encoder.
    """

    name = "instructblip"

    def __init__(
        self,
        model_name: str = "Salesforce/instructblip-flan-t5-xl",
        device: str = "cuda",
    ):
        super().__init__(model_name=model_name, device=device)
        self.processor: InstructBlipProcessor | None = None
        self.model: InstructBlipForConditionalGeneration | None = None

    def prepare_inputs(self, image, question: str) -> dict:
        """Prepare one deterministic InstructBLIP image-question example.

        ``image`` may be a filesystem path, a ``PIL.Image.Image``, or ``None``
        for a no-image baseline. The raw question is passed to
        ``InstructBlipProcessor`` so both the Q-Former tokenizer and the T5
        tokenizer receive the same instruction text. Floating tensors are moved
        to the adapter device in fp16; integer tensors remain integer.
        """
        self._ensure_loaded()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        question = question.strip()
        zero_pixels = image is None
        image_size = int(getattr(self.model.config.vision_config, "image_size", 224))
        pil_image = (
            Image.new("RGB", (image_size, image_size), 0)
            if zero_pixels
            else self._load_image(image)
        )
        encoded = self.processor(
            images=pil_image,
            text=question,
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
            "Prepared InstructBLIP inputs: %s",
            {
                key: list(value.shape)
                for key, value in prepared.items()
                if torch.is_tensor(value)
            },
        )
        return prepared

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for InstructBLIP, but torch.cuda.is_available() is false"
            )

        self._seed_everything()
        processor = InstructBlipProcessor.from_pretrained(self.model_name)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model = model.to(self.device)
        model.eval()

        processor.num_query_tokens = int(model.config.num_query_tokens)
        self._ensure_image_token_id(processor, model)
        self.processor = processor
        self.model = model

    def _run_first_decode_step(self, inputs, capture: bool):
        self._ensure_loaded()
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key
            in {
                "pixel_values",
                "qformer_input_ids",
                "qformer_attention_mask",
                "input_ids",
                "attention_mask",
                "inputs_embeds",
            }
        }
        if not model_inputs:
            raise ValueError("inputs do not contain InstructBLIP model tensors")

        batch_size = self._batch_size(model_inputs)
        start_token_id = self.model.config.text_config.decoder_start_token_id
        if start_token_id is None:
            start_token_id = self.model.config.text_config.pad_token_id
        if start_token_id is None:
            raise ValueError("decoder start token id and pad token id are both undefined")
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

        first_step_logits = outputs.logits[:, 0, :]
        return outputs, first_step_logits

    def _capture_lm_encoder_embeddings(self, inputs):
        """Capture the exact embeddings InstructBLIP feeds to the T5 encoder."""
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("attribution inputs must contain batched input_ids")
        qformer_input_ids = inputs.get("qformer_input_ids")
        if not torch.is_tensor(qformer_input_ids) or qformer_input_ids.ndim != 2:
            raise ValueError("attribution inputs must contain batched qformer_input_ids")

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key
            in {
                "pixel_values",
                "qformer_input_ids",
                "qformer_attention_mask",
                "input_ids",
                "attention_mask",
                "inputs_embeds",
            }
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
                    "InstructBLIP did not pass [batch, sequence, hidden] "
                    "embeddings to the language-model encoder"
                )
            captured_embeddings.append(encoder_inputs.detach())

        encoder = self.model.language_model.encoder
        handle = encoder.register_forward_pre_hook(
            capture_encoder_inputs,
            with_kwargs=True,
        )
        try:
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
        return encoder_embeddings

    @staticmethod
    def _ensure_image_token_id(processor, model) -> None:
        token_id = getattr(model.config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(model.config, "image_token_index", None)
        if token_id is None:
            image_token = getattr(processor, "image_token", None)
            token_text = getattr(image_token, "content", None)
            if token_text is None and isinstance(image_token, str):
                token_text = image_token
            converter = getattr(getattr(processor, "tokenizer", None), "convert_tokens_to_ids", None)
            if token_text is not None and converter is not None:
                token_id = converter(token_text)
        if token_id is None or int(token_id) < 0:
            raise ValueError(
                "InstructBLIP config/processor must expose a valid image token id"
            )
        model.config.image_token_id = int(token_id)
        model.config.image_token_index = int(token_id)
