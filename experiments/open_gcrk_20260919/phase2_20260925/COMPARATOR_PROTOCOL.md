# Prospective comparator and identity gate

**Status:** protocol amendment for future process-state development; no model
has been trained under it. The historical E3R2 outcomes have already been
inspected and remain exploratory.

## Why this gate is needed

The current E3R2 host has 42 damage inputs and a scalar learned logit smoother.
W+Cin adds a zero-initialized `6 x 32` context map, or **192 parameters**. The
existing GCRK adds **610 parameters** when geography has 40 dimensions
(`4 x 40` for the geographic code, three `32 x 4` maps, two 32-vectors and two
scalars). These are source-level counts, not performance results. A process
graph compared only with W+Cin could therefore confound topology with extra
capacity.

The second risk is cohort reuse. The twelve-event E3R2 panel and its OUTER
outcomes have been inspected. It can support an INNER/FIT-only timing and
optimization pilot, but it cannot independently confirm a topology chosen
after those results were known.

## Locked ladder

Every pilot bundle must contain the same seed/fold cells for:

1. `W+Cin`;
2. `W+Cin+causal_summaries`;
3. `discounted_weather_static_vulnerability` (the Zhu-style control);
4. `static_fragility_mixture`;
5. `generic_bounded_path_state`;
6. `structured_process_graph`.

The last two arms must have exactly the same number of new trainable
parameters, the same initialization-pair identifier, weather/geography inputs,
residual-amplitude cap and stopping rule. All arms share panel, split, weather
information set, feature names, observation support, normalization, host source
and training-protocol identities. Parameter matching is necessary but not
sufficient: wall time, peak memory and effective optimization steps must also
be reported.

## Phase boundary

- **Pilot:** the exact historical E3R2 hashes may be used only with
  `scope=inner_fit_only`, seed 0, and no OUTER artifact. The purpose is timing,
  numerical stability, budget audit and freezing architecture choices.
- **Confirmatory:** requires a new weather-selected cohort and split locked
  before label or model-result inspection, five seeds `0..4`, and one evaluation
  after preprocessing, architecture and stopping rules are frozen. The gate
  explicitly rejects the historical E3R2 panel or split hashes.

`comparator_protocol_gate.py` validates a bundle manifest against
`COMPARATOR_PROTOCOL.json`. Passing does **not** prove that a cohort is fresh or
that the listed files exist; source manifests, hashes and human review still do
that. It only prevents an internally inconsistent bundle from being promoted.

Run the deterministic checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/open_gcrk_20260919/phase2_20260925/comparator_protocol_gate.py --self-test
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s experiments/open_gcrk_20260919/phase2_20260925 -p 'test_comparator_protocol_gate.py' -v
```

Before a real pilot or confirmation, write the full bundle manifest to ignored
`runs/`, run the gate, and record its SHA-256 plus the gate output in
`ATTEMPTS.md`. A failed gate is a stop condition, not a warning.
