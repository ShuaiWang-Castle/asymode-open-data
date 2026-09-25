# Post-hoc diagnostic: power of the W2 pre-registered test on W1

**These files are post-hoc diagnostics, not a test.** They were produced after the W1 audit had been read, to inform
the W2 design (PREREG_W2 amendment 2), with `audit_f1.py --single "C1:p_tw-1.5*one@48" --min-eff-events 8
--require-event-flip --power-target 0.045 --diagnostic-power --scale {raw,log}` on the W1 panel.

* The registered feature is not eligible on W1 (effective clusters: event x state 13.6, county 70, event 3.5), so the
  test itself was not computed; `--diagnostic-power` only evaluates power.
* Power at part correlation 0.045 (half of W1's descriptive value), with noise built from W1's residual marginal
  properties: raw scale 0.34-0.53, log scale 0.23-0.40 across the four synthetic-noise variants.
* Reading: a feature as concentrated as this one on a panel of W1's size is underpowered at that effect size, and the
  log scale does not help. W2's registered criteria were kept unchanged (amendment 2).

Files: `H1a_*_{raw,log}.*` as written by the script (the `verdict` fields read "diagnostic power only").
