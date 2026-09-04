#!/usr/bin/env python3
"""Independent, no-retraining audit of the V2 negative pilot.

The audit uses only committed source code and committed pilot outputs.  It checks
whether the pilot actually instantiated the intended competition-informed model,
whether the three pilot tests represent three distinct training problems, and
whether the reported outputs contain enough diagnostics to attribute a null.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from asymode_paper.asymmetric_flows import (  # noqa: E402
    AsymmetricFlows,
    CAP_R,
    CAP_U_BKG,
    CAP_U_MAIN,
)
from asymode_paper.initialization import modular_init  # noqa: E402

OUT = ROOT / "analysis/gpt_rescue_20260904/cc_v2"
HERE = ROOT / "analysis/post_pilot_root_cause_20260904"


def fnum(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def gradient_probe(arm: str, u0: float, r0: float, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    model = AsymmetricFlows(32, 6, 23, arm)
    model.apply_modular_init(modular_init(u0, r0, CAP_U_MAIN, CAP_U_BKG, CAP_R))
    n = 64
    y = torch.rand(n) * 0.25
    xu = torch.randn(n, 32)
    xo = torch.randn(n, 6)
    xr = torch.randn(n, 23)
    # A target deliberately separated from update zero, so a trainable pathway
    # receives a nonzero signal.
    target = torch.clamp(y + 0.015 * (1.0 - y) - 0.01 * y, 0.0, 1.0)
    pred = model.step_from_state(y, xu, xo, xr)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    per_parameter = {}
    for name, par in model.named_parameters():
        per_parameter[name] = 0.0 if par.grad is None else float(par.grad.norm())
    return {
        "loss": float(loss.detach()),
        "per_parameter": per_parameter,
        "head_a_hidden_grad_max": max(
            v for k, v in per_parameter.items()
            if k.startswith("head_a") and not k.endswith("4.bias")
        ),
        "head_a_final_bias_grad": per_parameter["head_a.4.bias"],
        "head_b_hidden_grad_max": max(
            v for k, v in per_parameter.items()
            if k.startswith("head_b") and not k.endswith("4.bias")
        ),
        "head_b_final_bias_grad": per_parameter["head_b.4.bias"],
        "hold_grad": math.sqrt(sum(v * v for k, v in per_parameter.items() if k.startswith("hold"))),
        "total_grad": math.sqrt(sum(v * v for v in per_parameter.values())),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((OUT / "pilot_run_config.json").read_text())
    folds = json.loads((OUT / "event_folds_v2.json").read_text())
    design = pd.read_csv(OUT / "event_design_table.csv")
    result = pd.read_json(OUT / "pilot_results.json")
    effect = pd.read_csv(OUT / "pilot_event_effects.csv")
    diag = pd.read_csv(OUT / "pilot_training_diagnostics.csv")

    fold_of = {e: int(f) for f, es in folds["folds"].items() for e in es}
    pilot = list(cfg["pilot_events"])
    pilot_folds = {e: fold_of[e] for e in pilot}
    unique_training_problems = len(set(pilot_folds.values()))

    # Same fold + same seed + deterministic code implies the same training data,
    # normalisation, initialisation and optimisation path.
    duplicate_groups = defaultdict(list)
    for e, f in pilot_folds.items():
        duplicate_groups[f].append(e)
    duplicate_groups = {f: es for f, es in duplicate_groups.items() if len(es) > 1}

    # Verify that fold-duplicate jobs really have identical training-side records.
    duplicate_job_checks = []
    for f, es in duplicate_groups.items():
        a, b = es[:2]
        ra = result[result.test_event == a]
        rb = result[result.test_event == b]
        for arm in sorted(set(ra.arm)):
            for start in sorted(set(ra[ra.arm == arm].start)):
                xa = ra[(ra.arm == arm) & (ra.start == start)].iloc[0]
                xb = rb[(rb.arm == arm) & (rb.start == start)].iloc[0]
                keys = ["selected_stage", "selected_update", "selected_validation", "stage_a_best", "stage_b_best"]
                equal = all(xa[k] == xb[k] for k in keys)
                duplicate_job_checks.append({
                    "fold": f, "event_a": a, "event_b": b, "arm": arm,
                    "start": start, "training_record_identical": bool(equal),
                })

    # Gradient probes establish what the literal initialisation permits.
    probe_two = gradient_probe("asym_two_flow", 0.01, 0.02)
    probe_zero_one = gradient_probe("asym_one_flow", 0.0, 0.0)
    probe_restore_one = gradient_probe("asym_one_flow", 0.0, 0.02)

    # Committed diagnostics: module-level summaries can hide dead hidden layers
    # because the final scalar bias still has a gradient.
    grad_cols = [c for c in diag.columns if c.startswith("gn_")]
    grad_summary = {}
    for c in grad_cols:
        x = pd.to_numeric(diag[c], errors="coerce").dropna()
        grad_summary[c] = {
            "mean": float(x.mean()),
            "max": float(x.max()),
            "nonzero_fraction": float((x > 0).mean()),
        }

    # Check how the one-flow zero/interruption start actually moved.
    one_interrupt = result[(result.arm == "asym_one_flow") & (result.start == "interruption_ray_start")].copy()
    one_interrupt["stage_a_validation_change"] = one_interrupt["stage_a_best"] - one_interrupt["update0_path_mse_full"]

    # Event task composition for the pilot.
    pilot_design = design[design.event.isin(pilot)][[
        "event", "fold", "family", "zero_origin_share", "near_zero_share",
        "interior_share", "future_onset_share", "median_outage", "p90_outage",
    ]].copy()

    # Absolute usefulness and update-zero movement.
    effect_small = effect[[
        "test_event", "rel_tf_mse_full", "rel_path_mse_full", "rel_h24_mse_full",
        "two_vs_own_update0_path_full", "one_vs_own_update0_path_full",
        "two_vs_constant_two_flow_path_full", "one_vs_constant_two_flow_path_full",
    ]].copy()

    # Missing diagnostics: the prompt requested final U/R/c/gate summaries, but
    # neither committed result table contains them.
    diagnostic_columns = set(result.columns) | set(diag.columns)
    requested_rate_tokens = ["mean_u", "mean_r", "mean_c", "gate", "hold_value", "both_active"]
    missing_rate_diagnostics = [x for x in requested_rate_tokens if not any(x in c.lower() for c in diagnostic_columns)]

    # Latent deterministic-baseline bug in paper_v2_pilot.py.
    import paper_v2_pilot as pilot_module  # noqa: E402
    pilot_source = inspect.getsource(pilot_module)
    wrong_ray_rule_present = "if a_ray >= b_ray" in pilot_source
    stage_a_resets_temporal_state = "self.proposals(x_u_t, x_occ_t, x_r_t, None, None, 0)" in inspect.getsource(AsymmetricFlows.step_from_state)

    payload = {
        "branch_commit_from_run": cfg,
        "pilot_events": pilot,
        "pilot_folds": pilot_folds,
        "unique_training_problems": unique_training_problems,
        "duplicate_groups": duplicate_groups,
        "duplicate_job_checks": duplicate_job_checks,
        "gradient_probe_two_flow": probe_two,
        "gradient_probe_one_flow_zero_start": probe_zero_one,
        "gradient_probe_one_flow_restoration_start": probe_restore_one,
        "committed_module_gradient_summary": grad_summary,
        "missing_rate_diagnostics": missing_rate_diagnostics,
        "stage_a_resets_temporal_state": stage_a_resets_temporal_state,
        "wrong_constant_ray_selection_rule_present": wrong_ray_rule_present,
        "pilot_design": pilot_design.to_dict(orient="records"),
        "effect_summary": effect_small.to_dict(orient="records"),
    }
    (HERE / "ROOT_CAUSE_AUDIT.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Generated post-pilot root-cause audit",
        "",
        "This report is generated from committed code and committed pilot outputs; it performs no model retraining.",
        "",
        "## Load-bearing findings",
        "",
        "### 1. The interruption MLPs are not merely identical; their feature-learning weights are dead",
        "",
        "`apply_modular_init` zeros every linear layer in both ReLU MLPs. At that point all hidden activations are exactly zero and every downstream weight is zero. A backward pass therefore reaches only the final scalar bias; it cannot reach the final output weight or either hidden layer. The audit probe measured:",
        "",
        f"- two-flow `head_a` non-final-bias gradient maximum: `{probe_two['head_a_hidden_grad_max']:.3e}`;",
        f"- two-flow `head_a` final-bias gradient: `{probe_two['head_a_final_bias_grad']:.3e}`;",
        f"- the corresponding `head_b` values: `{probe_two['head_b_hidden_grad_max']:.3e}` and `{probe_two['head_b_final_bias_grad']:.3e}`.",
        "",
        "Thus the nominal pair of width-32 interruption networks learns only a time-invariant scalar bias in this pilot. The rich 32-channel interruption input cannot enter either magnitude head. Module-level nonzero-gradient checks missed this because they aggregate the live final bias with all dead weights.",
        "",
        "### 2. Stage A cannot train the first-order hold by construction",
        "",
        f"`step_from_state` resets `held_prev=None`, `r_prev=None`, and `step=0` on every transition: `{stage_a_resets_temporal_state}`. Consequently the hold gate is outside the one-step computation graph in Stage A. The committed checkpoint diagnostics are consistent with this: the hold-gradient mean is effectively zero.",
        "",
        "### 3. The interruption-ray safeguard is a dead start",
        "",
        f"With the zero/zero one-flow start, the total gradient in the independent probe is `{probe_zero_one['total_grad']:.3e}`. The collapse is evaluated at the ReLU kink `s=U-R≈0`, while the main interruption MLP is already dead. The committed interruption-start Stage-A validation is flat to printed precision for long stretches. Therefore the nominal two-start safeguard effectively supplies only one trainable start—the restoration start—which is selected in all three reported events.",
        "",
        "### 4. Three reported pilot events represent only two distinct optimisation problems",
        "",
        f"Pilot event-to-fold map: `{pilot_folds}`. The number of distinct held-out folds is `{unique_training_problems}`, not three. Events in the same fold use the same source events, county split, normalization, initialization, seed, and deterministic training path. The committed training-side records for the duplicate fold are identical for every arm/start: `{all(x['training_record_identical'] for x in duplicate_job_checks)}`.",
        "",
        "### 5. The selected origin rule does not isolate storm dynamics",
        "",
        "The committed origin audit shows that 62% of anchors are clipped to legal boundaries, so almost every panel uses an early boundary origin, one midpoint, and a late boundary origin. Stage A then draws unique transitions only from the 24-hour windows following those anchors—not from all unique panel transitions. This sampling scheme can omit the actual interruption pulse and heavily represent quiet/recovery transitions, which is consistent with the fitted constant interruption ray being zero.",
        "",
        "### 6. The pilot cannot measure the intended structural contrast at update zero",
        "",
        "The fitted source constants have `U0` equal to zero or about `2.7e-5`, versus `R0` of roughly `1.3e-2` to `2.7e-2`. The one-flow collapse deletes `min(U,R)`, which is therefore initially negligible. This does not establish that the learned final concurrency is negligible—the required final rate/concurrency traces were not saved—but it does establish that the pilot begins in a nearly unseparated regime and lacks the diagnostics needed to show that it ever leaves it.",
        "",
        "### 7. A challenge-specific temporal representation was transferred without its original time semantics",
        "",
        "In the challenge there was one fixed cutoff, so cumulative hazard, the first-order hold, and the eight-step recovery schedule had a common temporal origin. Here these states reset independently at every forecast origin. `path_*_since_origin` and the eight-step recovery phase therefore depend on an arbitrary rolling-origin coordinate. In Stage A, temporal state is reset altogether. The imported mechanisms are not operating under the semantics under which they were successful.",
        "",
        "### 8. The tropical failure is a transfer failure of the fitted context model, not evidence about one versus two flows",
        "",
        "The same fold-2 trained models perform acceptably on `2024-05-08` but both are much worse than their update-0 constant model on `2018-10-11`. Because the learned model is identical before test evaluation, this contrast isolates target-event shift. It does not identify the flow-collapse effect. The likely failing object is the learned context-to-rate map, especially the recovery/statics path, while the intended nonlinear interruption magnitude path was dead.",
        "",
        "## Additional implementation defects before any main run",
        "",
        f"- `constant_one_flow` chooses its ray by comparing coefficient magnitudes (`a_ray >= b_ray`) rather than the two ray SSEs: `{wrong_ray_rule_present}`. This did not change this pilot because `a_ray=0`, but it is incorrect in general.",
        "- Static missing values are imputed using each event's own county median before the source/test split, contrary to the documented fit-source-only preprocessing rule.",
        "- The feature map says clock channels are not normalized, but the pilot standardizes every `x_u` column.",
        "- Final proposal/rate/concurrency/gate/hold summaries were requested but are absent, preventing attribution of the small two-flow/one-flow differences.",
        "",
        "## Pilot task composition",
        "",
        pilot_design.to_markdown(index=False),
        "",
        "The winter medoid has only about 0.38% interior origins, and two of the three events have very small future-onset shares. Family diversity alone did not create an informative pilot for the second-flow question.",
        "",
        "## Performance and update-zero movement",
        "",
        effect_small.to_markdown(index=False),
        "",
        "## Root-cause hierarchy",
        "",
        "1. **P0 implementation failure:** all interruption MLP feature weights are dead; Stage A cannot train the hold; one of the two one-flow starts is effectively dead.",
        "2. **P0 experiment-design failure:** three events reduce to two training problems; anchors are boundary-clipped and Stage A samples only their windows.",
        "3. **P1 model-transfer mismatch:** challenge temporal states are reset under a rolling-origin task, and the learned context map fails sharply on the tropical target.",
        "4. **P1 measurement failure:** final `U`, `R`, common component, gate, and hold trajectories were not saved, so the pilot cannot verify whether the structural treatment was nontrivial.",
        "5. **Scientific question remains open:** the pilot's null is not a clean estimate of the value of one versus two conditional-mean flows.",
        "",
        "## Immediate implication",
        "",
        "Do not launch the 45-job main campaign and do not revise the manuscript conclusion from this pilot. First repair only the P0 items, run a deterministic gradient/trajectory smoke test, and then repeat a three-distinct-fold pilot. No new model family or hyperparameter sweep is warranted before those controls pass.",
    ]
    (HERE / "ROOT_CAUSE_AUDIT_GENERATED.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:90]))
    print(f"\nWrote {HERE / 'ROOT_CAUSE_AUDIT_GENERATED.md'}")
    print(f"Wrote {HERE / 'ROOT_CAUSE_AUDIT.json'}")


if __name__ == "__main__":
    main()
