# CrossRoute-Audit

CrossRoute-Audit is a framework for auditing the **faithfulness of explanation
methods** on white-box vision-language models. It checks whether an attribution
map ranks the modalities and token groups that actually drive a target output,
as measured by causal intervention.

The central question: *do attribution and causal intervention effect agree on
which modality or token group matters?* The primary metric is **rank alignment**
(Spearman correlation), which is invariant to differences in scale between
attribution scores and logit effects.

## Layout

```
crossroute_audit/
  model_adapters/   base.py (required interface) and blip2_adapter.py
  instrumentation/  attention capture, activation cache, hooks (no-op control)
  interventions/    ablation, activation patching
  controls/         text-only / no-image / counterfactual baselines (gating)
  attribution/      integrated gradients (primary), completeness, method agreement
  metrics/          rank_alignment (primary), causal_effect, attribution_mass,
                    flow_diagnostics (secondary), diagnosis (control-gated)
  io/               manifest loading, schema validation, reporting
  dashboard/        read-only viewer over JSON artifacts (later phase)
  cli.py            audit / batch / validate / report
schemas/            manifest.schema.json, audit_report.schema.json
data/manifest/      samples.example.jsonl
synthetic/          synthetic_faults.yaml
configs/            default.yaml
tests/              test_no_op_control.py
```

## Getting started

```bash
python -m pip install -e .

# Validate the example manifest
python -m crossroute_audit.io.manifest data/manifest/samples.example.jsonl

# Target CLI (available after the diagnosis milestone)
# crossroute audit --model blip2 --image data/images/dog_park.jpg \
#   --question "Is there a dog?" --target yes \
#   --controls text_only,no_image,counterfactual --out runs/dog.json
```

## Specification

The full MVP specification (task, target policy, interventions, metrics, control gates,
artifacts, CLI) is in [`docs/SPEC.md`](docs/SPEC.md).

## Design principles

- Rank alignment is the primary metric; the attribution-flow gap is secondary
  and reported only with an explicit normalization.
- Text-only/no-image and counterfactual controls gate any false-attribution
  flag; they are prerequisites, not appendices.
- A primary mismatch must hold across at least two attribution methods on a
  subset, otherwise confidence is lowered.
- Claims are about explanation faithfulness. Layer-wise statements apply only to
  white-box models.
- Scope is locked to a single model (BLIP-2) and one fault class for the first
  milestone; additional models and a polished dashboard come later.
