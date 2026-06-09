# CrossRoute-Audit — MVP Specification

Status: locked for the MVP sprint. This is the technical contract; positioning and
venue strategy are tracked separately.

## Objective
Test whether a vision-language model's attribution is *faithful* to the causal effect
of modality/token routes on a target output. Run end-to-end on one model, one task
family, with control gates, synthetic validation, and reproducible JSON artifacts.

## Scope
- **Model (MVP):** BLIP-2 (flan-t5-xl) — Q-Former/cross-attention gives a clear
  intervention surface. Adapter-based, model-agnostic core (LLaVA/Qwen-VL are phase 2).
- **Task:** VQA / object-presence with clear visual dependency
  (e.g. "Is there a dog?", "What color is the car?"). No long open-ended generation.
- **Audit unit:** a single target token/logit, not the full answer.
- **Scale:** 50–100 samples to debug; 20–30 clean samples for the workshop report.

## Manifest (one line per sample)
`sample_id, image_path, question, target_answer, target_token_policy,
expected_visual_dependency, text_only_answerable, counterfactual_image_path,
control_type, notes`. Validated by `schemas/manifest.schema.json`.
Target policies: exact answer token / first generated token / selected candidate logit.

## Interventions (causal anchor)
Ablate image route → `C_image_l`; ablate text route → `C_text_l`; clean-corrupt
activation patching → `patch_effect_l`; text-only/no-image baseline; counterfactual
image substitution; negative-control ablation (expect ~0). Every hook ships a no-op
control that must leave the clean target logit unchanged.

> Audit note (C): activation patching is the stronger causal probe; pure ablation is
> a supporting signal only, because zeroing a route can push the model off-distribution.

## Attribution
Layer Integrated Gradients on the chosen target logit, **computed at layer l's
activations** (primary). A second method (attention rollout / grad saliency) on a small
subset for method agreement. Report baseline, steps, target token, grouping, completeness
residual; high residual or method disagreement ⇒ low-confidence.

## Metrics
- `CausalEffect_l = target_logit_clean − target_logit_intervened`.
- `AttributionMass_l` = attribution summed over an activation group at layer l.
- `RankAlignment_l` = Spearman ρ between `AttributionMass_g,l` and `CausalEffect_g,l`
  across groups g — **primary metric** (scale-invariant).
- Secondary: `AttributionFlowGap_l = |A_l − C_l|` after explicit normalization;
  `FlowRetention_l = C_image_l / C_image_early`; `EffectStability_l` across seeds.

> Audit note (A+B): RankAlignment is computed over groups defined at sufficient
> granularity — partition image tokens into K spatial regions plus content text tokens
> so that **n_groups ≳ 10–20 per layer**; never just the 3 modality groups, or Spearman is
> degenerate. AttributionMass and CausalEffect must share the same layer l and the same
> activation groups. Report `n_groups`; flag small-n layers as low-confidence.

## Control gates (a case is flagged "False Attribution Persistence" only if ALL hold)
text-only/no-image cannot explain the target; counterfactual image was checked; negative
controls clean; target logit stable; no-op intervention leaves output unchanged; ≥2
attribution methods agree on the subset; attribution residual not too high.
Do not flag if the model answers via language prior, or if the counterfactual does not
flip the target as expected.

## Diagnosis flags
False Attribution Persistence; Modality Drop (low causal effect of the expected modality
across layers); Route Break (routing proxy and causal effect both drop near a layer).
No serious diagnosis when logit is unstable / residual high / a control fails.

## Artifacts
`sample_manifest.json, control_status.json, flow_graph.json, causal_effect.json,
attribution_mass.json, audit_report.json, benchmark_summary.csv`. Aggregate only — no raw
tensors in JSON.

## CLI
```
crossroute audit    --model blip2 --image sample.jpg --question "Is there a dog?" \
                    --target yes --controls text_only,no_image,counterfactual --out result.json
crossroute batch    --manifest samples.jsonl --out runs/mvp_blip2/
crossroute validate --suite synthetic_faults.yaml --out validation/
crossroute report   --run validation/ --out report.md
```

## Non-goals (MVP)
Multiple models, polished dashboard, multiple attribution methods at scale, long-form
generation, journal-scale evaluation. These are phase 2.

## Claim boundary
Layer-wise claims apply only to white-box models with hook/activation access. The claim is
an audit of explanation-method faithfulness — not that the model reasons incorrectly.
