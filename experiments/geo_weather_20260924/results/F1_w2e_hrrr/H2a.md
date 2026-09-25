# Registered single test C8:p_tw-1.5*one@0 (through audit_f1_hrrr.py)

Generated 2026-09-25 06:51:55 (git 3c17935; audit_f1.py sha 72ebcfcdaa3b…, wrapper sha 6cb852edd1e0…). Panel `features_w2e.npz`, base runs/geo_weather_20260924/w2e_base/fold{fold:02d}/outer.npz. Order: eligibility from the design, then power (written by the frozen function to `H1a_power.json` before the test), then the test.

* Units 10971, events 23, event x state clusters 635; nonzero contrast in 596 blocks and 23 events.
* Exclusions: unit_ok false 1843, excluded events none (0 units), units without a valid hour 0; largest missing share of observed hours in a kept event 0.042.
* Effective clusters: event x state 47.1, county 463.0, event 12.0; required ≥ 20 / ≥ 20 / ≥ 8.0: **eligible**.
* Single-test false-positive rate on synthetic fields: homo l 0.059, homo 2l 0.043, hetero l 0.073, hetero 2l 0.066; null used: cluster multiplier.
* Power at part correlation 0.117: homo l 0.996, homo 2l 0.998, hetero l 0.840, hetero 2l 0.838; minimum 0.838 (**informative**).
* Test: part corr +0.0702, t (event x state) +3.11, p multiplier 0.0005, p synthetic 0.0010, p used 0.0005; t (event) +1.94, exact event-flip p 0.0040.

**Verdict: pass.**

