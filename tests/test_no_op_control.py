"""No-op control test: the causal-correctness gate.

A no-op hook must leave the clean target logit effectively unchanged. This test
runs once the BLIP-2 adapter implements ``intervene(mode="noop")``.
"""
import pytest

pytest.skip(
    "Enable once BLIP2Adapter.intervene(mode='noop') is implemented.",
    allow_module_level=True,
)


def test_noop_preserves_target_logit():
    # clean = adapter.get_target_logit(...)
    # noop = adapter.intervene(..., mode="noop")
    # assert abs(clean - noop) < 1e-4
    ...
