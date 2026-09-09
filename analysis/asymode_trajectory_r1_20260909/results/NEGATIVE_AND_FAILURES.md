# Negative results, failures and things not run

## Failed / aborted invocations, kept on the record

1. **First `--phase main` launch exited 127.** A zsh quoting fault: the command was
   stored in a shell variable and expanded as a single word. Nothing trained, no
   results were produced, no trial was deleted. Relaunched verbatim as a direct
   invocation. Recorded because §5 forbids silently discarding failed trials.
2. **First `prepare_local_data.py` run raised `BLOCKED_DATA`.** The project working
   directory `D` does not contain the ANEEL corpus. Resolved by passing the
   additional local root `DMDA/ANEE`, as §1 permits. No cohort, denominator or
   hour boundary was changed, and no deleted ZIP was requested.
3. **An ad-hoc `legal_origins` call during profiling returned 3768 origins** instead
   of the locked 3745. Cause: hand-passed dates rather than the frozen
   `data_contract.PARTITIONS` tuple. The profile was informational only; every
   formal run uses the preflight-frozen origin file, and the driver refuses to
   start unless all six frozen origin sets reproduce exactly from `PARTITIONS`.

## Negative and unfavourable findings, retained

- **NET is not reliably better than DIRECT.** Seed 5103 gives a *negative* mean
  company relative gain (-0.287%); only 4 of 5 paired seeds favour NET.
- **ASYM is worse than both baselines at the 1-hour endpoint.** 1.125e-04 versus
  DIRECT 1.090e-04 and NET 1.082e-04. Its advantage appears only at longer
  horizons. Reported rather than dropped in favour of the endpoint that flatters it.
- **The real-data effect is small.** ASYM's mean-company-relative gain over DIRECT
  is +1.24%, and the descriptive 168h block interval for ASYM (+0.29%, +1.24%)
  overlaps NET's (+0.24%, +0.82%). The two structured arms are not separated by
  this sensitivity analysis.
- **`ASYM_STEP_ONLY` is not worse than full-path ASYM** on 2019 (1.6902e-04 vs
  1.6904e-04). The locked full-trajectory training objective bought essentially
  nothing here. This is an unfavourable result for the training-objective part of
  the story and is reported as such.
- **On the controlled source split, NET beats ASYM** (3.18e-08 vs 9.18e-08; ASYM
  better in only 1 of 5 seeds). ASYM's controlled advantage is confined to the
  shifted-`Y0` split.

## Not run in this cycle, by protocol

`DN_READ`, `DN_PROJ`, AFF competition, direction-label supervision, source-log
correction, Gamma neural thresholds, low-state retraining campaigns, weather
components, intervention identification, certificates, report compression. No
new projection experiment was run; the historical projection counterevidence is
copied unchanged into `appendix_evidence/`.

## Known limitations of this execution

- No CUDA device exists on this host. The §6 reference comparison is FP32 compute
  against an FP64 CPU reference of the identical model, and `cuda_available` is
  reported `false` everywhere rather than claimed.
- Wall-clock, memory and inference cost differ across the three arms; the recursive
  arms are materially slower per update. Parameter and gradient-sample budgets are
  matched, FLOPs and walltime are not, and this run does not claim they are.
- Five paired seeds are replicates of one retrospective evaluation, not five
  independent field samples. 2019 has been used repeatedly and is labelled
  `REUSED_2019_RETROSPECTIVE_EVALUATION` throughout.
