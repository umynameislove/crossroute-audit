"""Architecture-agnostic model adapter contract.

The core audit engine must never depend on model-specific module names or
generation mechanics. Every adapter exposes the same token groups, audit layer
axis, target-logit path, attribution taps, routing proxies, and intervention
operations through this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenGroups:
    """Token positions grouped by audit role.

    Positions are zero-based indices in the sequence consumed by the adapter's
    audit stack. ``image`` and ``text`` are the canonical groups used by shared
    metrics. ``fusion`` should cover valid positions used for cross-route
    analysis, and ``answer`` identifies answer-token positions when applicable.
    """

    image: list[int] = field(default_factory=list)
    text: list[int] = field(default_factory=list)
    fusion: list[int] = field(default_factory=list)
    answer: list[int] = field(default_factory=list)


@dataclass
class ForwardOutput:
    """Tensor-free result of a forward pass.

    ``target_logit`` is a finite scalar selected by the adapter. ``hidden_states``
    and ``attentions`` may contain summaries only; raw tensors must not be
    stored in this dataclass or serialized artifacts.
    """

    target_logit: float
    hidden_states: Any = None
    attentions: Any = None
    meta: dict = field(default_factory=dict)


class ModelAdapter(ABC):
    """Contract every auditable model adapter must implement.

    The audit layer index is always zero-based and valid in
    ``0..get_intervention_layer_count()-1``. Canonical token-group names are
    ``image`` and ``text``. Adapters may expose extra groups in ``TokenGroups``,
    but shared metrics rely on those canonical names.
    """

    name: str = "base"

    @abstractmethod
    def prepare_inputs(self, image, question: str) -> dict:
        """Return deterministic model inputs for one image-question example.

        ``image`` is adapter-defined input data such as a path, image object, or
        ``None`` for the adapter's no-image baseline. ``question`` must be a
        non-empty string. The returned mapping must contain all tensors and
        metadata needed by later adapter calls, and repeated calls with the same
        arguments must produce equivalent inputs.
        """

    @abstractmethod
    def forward(self, inputs, capture: bool = False) -> ForwardOutput:
        """Run a clean first-step forward pass.

        ``inputs`` must be a mapping produced by :meth:`prepare_inputs`.
        Implementations return a finite scalar ``ForwardOutput.target_logit``.
        When ``capture`` is true, implementations may include tensor-free
        summaries of internal states; raw tensors must not be returned.
        """

    @abstractmethod
    def get_token_groups(self, inputs) -> TokenGroups:
        """Return image/text/fusion/answer positions for the audit stack.

        The returned positions must index the same sequence used by attribution
        and intervention calls. ``image`` and ``text`` must be non-overlapping
        lists of integers when both modalities are present. Implementations must
        validate sequence shape and fail loudly on inconsistent inputs.
        """

    @abstractmethod
    def get_layer_count(self) -> int:
        """Return an adapter-specific public layer count, if one exists.

        Shared audit code must use :meth:`get_intervention_layer_count` instead
        of this method. This method is kept for backward-compatible adapter
        metadata and smoke tests.
        """

    @abstractmethod
    def get_intervention_layer_count(self) -> int:
        """Return the length of the ordered audit layer axis.

        The audit layer axis is the sequence of layers where the adapter can
        both tap attribution and apply interventions while image/text routes are
        represented in a shared token space. Valid layer indices are
        ``0..get_intervention_layer_count()-1``.
        """

    @abstractmethod
    def layer_axis_name(self) -> str:
        """Return a non-empty adapter-defined name for the audit layer axis.

        The value is written into artifacts for reproducibility. It may be a
        module path or another stable logical name, but shared code treats it as
        an opaque string.
        """

    @abstractmethod
    def get_routing_proxy(self, inputs, layer: int):
        """Return an aggregated route-interaction proxy for one audit layer.

        ``layer`` must be a valid audit layer index. The return value must be a
        finite scalar or tensor-free scalar-like value; raw attention or hidden
        tensors must not be returned.
        """

    @abstractmethod
    def intervene(self, inputs, layer: int, group: str, mode: str):
        """Apply one route intervention and return the intervened target logit.

        ``layer`` must be a valid audit layer index. ``group`` uses canonical
        names such as ``image`` or ``text``. ``mode`` is one of {ablate, mask,
        patch, shuffle, noop}. ``noop`` is a required control and must leave the
        clean target logit effectively unchanged.
        """

    @abstractmethod
    def get_target_logit(self, inputs, target_answer: str, policy: str) -> float:
        """Resolve and return the finite scalar target logit.

        ``policy`` defines how ``target_answer`` maps to one target token.
        Implementations should store enough target metadata in ``inputs`` for
        later :meth:`intervene` calls to measure the same token.
        """

    @abstractmethod
    def prepare_attribution_inputs(self, inputs, target_answer: str, policy: str):
        """Return differentiable attribution inputs for one sample.

        Returns ``(embeddings, attention_mask, target_token_id)`` where
        ``embeddings`` has shape ``[1, sequence, hidden]``, ``attention_mask``
        has shape ``[1, sequence]``, and ``target_token_id`` is an ``int``. The
        embeddings must correspond to the same audit-stack sequence indexed by
        :meth:`get_token_groups`.
        """

    @abstractmethod
    def attribution_baseline_embeddings(self, inputs):
        """Return an on-manifold attribution baseline embedding tensor.

        The tensor must have the same shape, device, and sequence semantics as
        the embeddings returned by :meth:`prepare_attribution_inputs`. It must
        be deterministic and must not require gradients.
        """

    @abstractmethod
    def forward_target_logit_from_embeddings(
        self,
        embeddings,
        attention_mask,
        target_token_id: int,
    ):
        """Return a differentiable first-answer-step target logit.

        ``embeddings`` has shape ``[batch, sequence, hidden]`` and
        ``attention_mask`` has shape ``[batch, sequence]``. The returned tensor
        must contain one finite scalar logit per batch item. For
        autoregressive-only adapters, the target is the next token after the
        prompt; for adapters with a separate answer step, it is the first token
        generated for the answer.
        """

    @abstractmethod
    def attribution_layer_output(self, layer: int):
        """Context manager exposing one audit-layer output to attribution code.

        ``layer`` must be a valid audit layer index. The yielded object must be
        acceptable to the attribution backend as the tapped layer/module. The
        context manager must remove all hooks in ``finally``.
        """

    @abstractmethod
    def attribution_float32(self):
        """Context manager for numerically stable attribution computation.

        Implementations may temporarily cast only the required attribution path
        to float32. They must restore dtype/gradient state in ``finally`` and
        must not change the normal inference dtype after the context exits.
        """

    @abstractmethod
    def run_controls(self, inputs, sample: dict) -> dict:
        """Run adapter-level controls and return tensor-free control status."""
