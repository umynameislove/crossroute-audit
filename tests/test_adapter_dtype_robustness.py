"""Dtype-robustness regression tests for the VLM adapters.

These guard two real failures surfaced when the new adapters first ran on a
7B model (transformers 4.57):

* Flan-T5 keeps select modules in float32 (``_keep_in_fp32_modules``), so the
  BLIP-2 / InstructBLIP language model holds a *mix* of dtypes; the old
  ``attribution_float32`` asserted a single dtype and crashed.
* Qwen2-VL emits NaN logits in float16 and must load in bfloat16.

The BLIP-2 tests run on CPU (no model download); the Qwen test only inspects a
class constant, so neither needs a GPU.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from crossroute_audit.model_adapters.blip2_adapter import BLIP2Adapter
from crossroute_audit.model_adapters.qwenvl_adapter import QwenVLAdapter


def _adapter_with_mixed_dtype_lm():
    """Return a BLIP2Adapter whose language model mixes float16 and float32."""
    adapter = BLIP2Adapter.__new__(BLIP2Adapter)        # bypass model loading
    language_model = nn.Module()
    language_model.half_weight = nn.Parameter(torch.zeros(3, dtype=torch.float16))
    language_model.full_weight = nn.Parameter(torch.zeros(3, dtype=torch.float32))
    model = nn.Module()
    model.language_model = language_model
    adapter.model = model
    adapter._ensure_loaded = lambda: None               # type: ignore[assignment]
    return adapter, language_model


def test_attribution_float32_handles_mixed_dtype_language_model():
    adapter, lm = _adapter_with_mixed_dtype_lm()
    before = [p.dtype for p in lm.parameters()]
    assert torch.float16 in before and torch.float32 in before

    with adapter.attribution_float32():
        assert all(p.dtype == torch.float32 for p in lm.parameters())

    assert [p.dtype for p in lm.parameters()] == before   # restored per-parameter


def test_attribution_float32_restores_mixed_dtype_on_error():
    adapter, lm = _adapter_with_mixed_dtype_lm()
    before = [p.dtype for p in lm.parameters()]
    with pytest.raises(RuntimeError, match="boom"):
        with adapter.attribution_float32():
            raise RuntimeError("boom inside attribution")
    assert [p.dtype for p in lm.parameters()] == before


def test_attribution_float32_rejects_non_floating_language_model():
    adapter = BLIP2Adapter.__new__(BLIP2Adapter)
    language_model = nn.Module()
    language_model.idx = nn.Parameter(
        torch.zeros(3, dtype=torch.long), requires_grad=False
    )
    model = nn.Module()
    model.language_model = language_model
    adapter.model = model
    adapter._ensure_loaded = lambda: None               # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="floating-point"):
        with adapter.attribution_float32():
            pass


def test_qwenvl_uses_bfloat16_compute_dtype():
    assert QwenVLAdapter.COMPUTE_DTYPE == torch.bfloat16
