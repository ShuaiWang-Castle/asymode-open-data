# Review: DATASET_DESIGN v1 (adversarial methods review, reviewer receipt)

2026-09-26. **Object:** `DATASET_DESIGN.md` v1 as registered (commit 02b78a8), read together with the uncommitted
amendment 1 (05:08 EDT) and the in-progress frame code (`panel_v1/*.py`, 04:51-05:20 EDT).

**Read:** the three reviews it integrates; `src/asymode/panel.py`; `open_gcrk_20260919/build_panel216.py`; `program.md`;
`PREREG_W2.md` up to amendment 8; `aris/EXPERIMENT_PLAN.md`; and the code the design depends on
(`src/asymode/gcrk_train.py`, `open_gcrk_20260919/build_features.py`, `scripts/build_county_statics.py`,
`panel_v1/build_frame.py`, `draw.py`, `used_windows.py`).

**Numbers:** counts marked [frame] come from the first outcome-blind frame build (`data/interim/panel_v1/systems.parquet`,
05:16 EDT, uncommitted). That build uses only VTEC, HURDAT2, customers, coverage and national collection timestamps.
Coverage counts come from `data/interim/eaglei_coverage_history.parquet`.

**Not done:** nothing was trained or downloaded, no outage value was read, and no draw existed during the review.

**Severity.**
* **Blocking:** as written, an outcome-derived quantity reaches the sealed tranche C or a registered test.
* **Before the draw:** must be settled before `draw.py` runs. Otherwise it is settled later by judgement, possibly after
  D's outcomes are seen.
* **Minor:** wording, diagnostics, or small effects.

## 0. Verdict

The core of v1 is sound:
* an outcome-independent warning frame;
* PPS draws that keep every π;
* C drawn first and never supplemented;
* pre-window observation gates;
* a regime-balanced estimand whose loss has the same form.

Amendment 1 already fixes the worst problem the build found: tropical warnings split off from their storm. What remains:

1. **One blocking item.** The host's county context includes a 2023 outage statistic (SAIDI). Every 2023 system in D
   and C therefore carries part of its own outcome in its inputs.
2. **Thirteen items to settle before the draw.** Most are one-line pins of sets and formulas. These ones change who is
   drawn, or how they are weighted:
   * the compound flag cannot see the case it exists for (ice, then wind, in the same county);
   * the audit supplement uses the wrong π;
   * the secondary estimand counts the frame expansion twice;
   * G2 by the year's minimum drops 55% of 2019 systems;
   * C's population is "the frame minus the used windows" and should be declared as such.
3. **Fourteen review suggestions** are neither adopted nor listed in §14 (section 4).

## 1. Blocking

| # | § | What goes wrong | Amendment text |
|---|---|---|---|
| B1 | §9.3, §10 (host W+Cin) | The host's six county-context inputs (`STATIC` in `open_gcrk_20260919/build_features.py`) include `log1p_saidi`. Every arm uses them, in `ctx` and in the recovery inputs. `scripts/build_county_statics.py` builds `log1p_saidi` from the first SAIDI column of EIA-861 **Reliability_2023**. In EIA-861's layout that column normally includes major event days, and with or without them it is computed from 2023 interruptions. A utility's 2023 SAIDI therefore contains the 2023 outages that are the target of every 2023 system, about one system in seven in the frame. Consequences: a 2023 C system enters the confirmatory test with a function of its own outcome in its inputs; 2023 D systems score optimistically on event folds; Stage 0's headline regimes inherit the bias. The registered host breaks §10's own rule ("any outcome-derived number on C"). | "On the designed panel, the host's county context takes SAIDI from EIA-861 2017, the last year before the frame, instead of 2023. No model input may be computed from outage records dated inside 2018-07-12..2025-12-31, other than the county-event's own prefix. Stage 0 runs with this input." |

## 2. Settle before the draw

| # | § | What goes wrong | Amendment text |
|---|---|---|---|
| S1 | §3.4(2), §4.2, §4.4(4), §9.1 | **Sets and sources are undefined.** "Shares a county" and "county sets" have no definition. With S1-only sets, a D system whose ring (S3) counties are also a C system's ring counties in the same days is not blocked, and the same county-hours land in both D and C. The used-window sources are also open: the text says "the event files of the earlier panels", and the statistical review names four panels. The repository also read the 26 main-line panels (`data/interim/PANEL_MANIFEST.json`), including 2018-10-11, 2020-08-04, 2020-10-29 and 2024-09-27, the landfalls of Michael, Isaias, Zeta and Helene. PPS on customers gives storms like these inclusion probabilities near one in C's small tropical frame. The in-progress code already does the right thing (`used_windows.py`: 125 windows including the 26 panels; `draw.py`: S1-S3 sets), but the text must say so. §9.1 is not coded yet. | "In §4.2 and §4.4(4), a system's counties are its S1, S2 and S3 counties in the frame file (before G3-G5 and before county sampling), and a used window's counties are its listed footprint. In §9.1 they are the sampled counties. In §3.4(2) they are the S1 counties. The used windows are every event in `experiments/open_gcrk_20260919/selected_events*.json` and `experiments/geo_weather_20260924/selected_events*.json`, plus every main-line `data/interim/panel_<day>.npz` (its fips and timestamps only), as compiled by `panel_v1/used_windows.py`." |
| S2 | §3.5, §4.3, §4.6 | **The compound flag misses same-county compounds.** It is computed from county labels, and a label keeps only the highest-priority regime of the county. A county under IS and then HW is labelled winter only, so the flag sees only spatial mixes. It cannot see "ice then wind", the case §4.3 doubles M_s for (physical review §3). [frame]: 934 eligible systems are compound, 816 of them convective or heavy-rain mixes. The audit's "at least 5 compound systems" therefore passes without a single load-then-trigger case. | "A non-TC system is compound if at least 20% of its S1 customers are in counties whose multi-label vector holds a regime other than the system's regime." (This includes every system the current rule flags.) |
| S3 | §4.6 | **Four problems in the coverage audit.** (i) It runs on "domain counties (§5.1)". Those need G3-G5, which come in step 4, but the audit is in step 3. (ii) The "deficient cell" of a geography tercile is not defined for systems, which span terciles. (iii) Unspecified: how many systems "up to 3" means, what happens when several conditions fail, and which regime supplements a D-wide compound failure. (iv) The formula π = 1 − (1 − π₁)(1 − π₂) treats the supplement as an independent second draw. In fact the supplement happens only when the audit fails, and it draws from systems not yet in D. Example: π₁ = 0.05, π₂ = 0.30, and the audit fails in 30% of replays. The true π ≈ 0.05 + 0.95 × 0.3 × 0.3 ≈ 0.14, while the formula gives 0.34. The weight is understated about 2.5 times, and only for the rare geographies the supplement exists to add. | "(i) The audit uses the S1-S3 counties that pass G1-G2 and are not operator-excluded. (ii) A system is in a deficient geography cell if at least half of those counties lie in the deficient tercile. (iii) Each failing condition, in the order relief, canopy, drainage, density, coast, season, compound, gets one PPS draw of 3 systems (fewer if the cell has fewer). The draw comes from the cell's D-frame systems not yet in D. A D-wide compound failure is supplemented in the regime whose D frame holds the most compound systems. A condition that still fails is recorded, and nothing more is drawn. (iv) Each system's D π is the share of 10,000 replays of the D draw, the audit and the supplements (uniform starts, realised C draw fixed) that include it. This replaces 1 − (1 − π₁)(1 − π₂)." |
| S4 | §6 G2 | **The year's minimum coverage is too blunt.** Lower-48 states passing 0.8 by the year's minimum / by the year's maximum: 2018 24/32, 2019 23/40, 2020 34/38, 2021 35/40, 2022 39/41. [frame]: among systems passing S-a to S-c, S-d removes 55% of 2019 systems and 16-18% in 2022-2025. The minimum is a single instant anywhere in the calendar year. A dip in March removes a state's September storms. A storm that takes a large utility's outage feed down can set that state's minimum and so remove itself, which lets the outcome decide membership. The literature note's recommendation 6.7 (an in-data check before the window) is neither adopted nor listed in §14. A county-level check cannot run at the frame step, because it would read future C windows. It fits at the county-sample step instead. | "S-d and M_s use the year's maximum coverage ≥ 0.8. At the county-sample step (step 4 for D; C's build for C), a state whose minimum is below 0.8 keeps its counties only if at least 80% of its 2024 modelled customers are in counties with a positive record in the 30 days before the window. That read is pre-window only, with the C mask of S8. The frame lists every state-year concerned." |
| S5 | §1, §4.2 | **C's population is the frame minus used systems.** The used windows were chosen from damage reports (the earlier panels were Storm Events selections), so removing them strips many of the largest 2018-2025 winter and wind storms, and four TC systems, out of C's frame. C then estimates "the frame minus used systems" while D, which keeps them, estimates another population. The statistical review warned that the sealed winter stratum would skew small unless size is stratified (§4). v1 neither stratifies by size nor lists the warning in §14. | "C's inference population is the frame minus the used systems. The frame file reports, per regime, the used share of systems and of M. D results shown beside C results are also reported on D minus used systems. §1's claim population for C says so." |
| S6 | §3.7 | **Drawn systems can lose every county.** Under the pre-window service rule, G4 and G5 vary by county only through blank rows. Otherwise they reduce to national collection runs in the prefix and at the origin. S-b allows up to 21 missing hours anywhere in the 216 h window. If those hours fall in the prefix or on the origin hour, every county fails G4 or G5 after the system has been drawn, and v1 has no replacement rule (C is never supplemented). | "S-e. National collection runs cover the origin hour and at least 90% of the prefix hours." |
| S7 | §1, §9.3 | **Estimand and loss are inconsistent in three places.** (a) The regime r of a county-event is undefined: the system's regime, or the county's label? S2 and S3 counties have no label. (b) The loss's R comes from the headline regimes, but Stage 0 must train before any headline regime exists. (c) The secondary estimand's "system weights N_h / n_h inside the design weights" counts the expansion twice. Under PPS, 1/π_s already sums in expectation to the regime's frame count, so the extra factor weights cell h by about N_h²/n_h instead of N_h. | "(a) In MSE_r, Z_r and the loss, a county-event belongs to its system's regime; labels are used for breakdowns only. (b) Every loss, Stage 0's included, averages over the five regimes (R = 5). A regime with Z_r = 0 in a run's training folds is left out of that run. The primary estimand averages over the headline regimes, so 'the loss has the same form' holds exactly when all five regimes are headline. (c) The frame-weighted estimand pools the regimes with the design weights w of §5.4. N_h enter only a post-stratified check with weights w · N_h / Σ_{sampled s in h} 1/π_s." |
| S8 | §6 G3, §10, §11 | **D-side reads can reach C's records** (fix before step 4). §4.4(4) blocks only D windows that share a county with a C window and start within 16 days of it. A D window that starts 16-39 days after such a C window (16-99 days with the 90-day lookback) is not blocked, and its G3 lookback reads records inside the C window. §10 counts "gate counts from forecast-window records" as a read of C. The §11 diagnostic removes "C's windows" without saying which county-hours. A fast G3 implementation, one county × day presence table for 2018-2025, would read every C window at once. And the G3 switch "reads no outcome": in fact it reads pre-window records, and the text does not say whether the switch applies to C and P. | "Every EAGLE-I read outside the confirmatory script (G3 of D, the §11 diagnostic, any audit) skips the county-hours of C windows, i.e. C's S1-S3 counties × [window start, window end + 7 d]. A lookback cut this way uses the hours that remain. No table spanning a C window is materialised. The G3 lookback switch is decided on D's pre-window records and then applies to every tranche." |
| S9 | §8 | **The HRRR mask changes the scored hours** (fix before Stage B). "Masked like unobserved outage hours" makes the set of scored county-hours depend on the weather source. A HRRR arm and its ERA5 twin, or the host, would then be scored on different hours unless they share one mask. H2b entered missing hours as zeros (PREREG_W2 amendment 6). | "Missing HRRR hours enter the HRRR inputs as zeros, as in H2b, and are listed. They never change the loss or metric mask." |
| S10 | §9.2, program.md rule 3b | **The designed panel has no keep or discard criterion** (fix before Stage B). Rule 3b replaces rules 1-3, and with them rule 2's threshold, interval and placebo test, but it states no keep threshold, no interval and no discard criterion. The keep line would therefore be drawn after results are seen. | "Keep: seed-averaged over three seeds on the five event folds, the regime-balanced gain exceeds 1% against the host and is positive against the twin. Each gain needs its 95% family-cluster bootstrap interval above zero, and no non-headline regime may lose more than 2%. Discard: a single-seed five-fold screen whose regime-balanced gain against the host is ≤ 0." |
| S11 | §10 | **"Spent" is undefined.** A second registration could use the rest of the K = 4 budget after the first one has revealed its statistics on C. | "One confirmatory registration per tranche. A tranche is spent when that registration's sequence stops." |
| S12 | §3.4-§5.1 | **Choices the build had to make; v1 leaves them open.** (1) TC systems are not segmented. [frame]: 13 of 57 TC systems have warnings ending more than 144 h after onset: Barry, Beryl, Debby, Delta, Dorian, Elsa, Fay, Florence, Gordon, Helene, Ida, Isaias, Sally. (2) S2 counties are ring counties: either touched by the system (a < 0.1) or under a listed A or Y product of any system in the forecast window. Advisory counties outside the ring are not in the domain. (3) A W event with no county at a ≥ 0.1 creates no system and acts as an advisory. (4) M_s leaves out operator-excluded counties, whereas §4.3 says G1-G2 only. (5) Typing and region use all S1 customers, ungated. The season class uses the origin month (UTC). Sort ties keep frame order. (6) S1 membership is not limited to products issued before the hazard (statistical revision 3; physical review §4 on zones). Zone extensions (EXA/EXB) count, so S1 partly reacts to damage reports. | "The frame rules are (1)-(5) as built in `panel_v1/build_frame.py` at the commit of the frame. Because (6) lets S1 react to damage reports, the stratum breakdown of §1 is descriptive, not a covariate breakdown. S1 is a sampling stratum with known π, so this changes the strata, not the unbiasedness of the domain estimand. §14 lists statistical revision 3 as not adopted, for that reason." |
| S13 | §3.9, §7, §4.4 | **Committed files cannot be committed.** `.gitignore` excludes `data/`, `*.parquet` and `*.csv`. The frame (`systems.parquet`, `system_counties.parquet`), `data_provenance/operator_exclusions.csv` and the tranche list (`draws.parquet`) therefore cannot be "committed before" anything. `draws.json` holds the seeds, sizes and frame hash, but not each system's tranche. Without a committed tranche list, no one can later verify that C was fixed before D's outcomes were read. | "The frame, the exclusions and the tranche list (system, regime, family, tranche, π) are committed as JSON under `data_provenance/` (or force-added), each with its SHA-256, at the steps §12 names." |

## 3. Minor

| # | § | Point | Amendment text or action |
|---|---|---|---|
| M1 | §3.4(2) | **Long events chain unrelated storms.** [frame]: 125 of 239,157 W events last more than 168 h: 99 FA, plus 5 TO and 1 SV that can only be record errors. The longest lasts 6,831 h. Yet every phenomenon's 99th-percentile duration is at most 144 h. These events chain 11 families over 21-92 days; the largest holds 17 systems in 4 regimes (2023-02-17 to 05-19, bridged by an 81-day FA.W). A C draw anywhere in such a family blocks the whole family from D. 73 eligible systems are affected. | "For linking, an event's interval ends at min(last EXPIRED, INIT_ISS + 168 h)." |
| M2 | §3.4 | **Same-storm families can stay apart.** A graph family linked in time and space to a TC system's events (for example TO or FF beyond 300 km) keeps its own family. The county-time rules then protect independence only through shared counties. | "A graph family linked by §3.4(2) to an event of a TC system joins that TC system's family, not its system." |
| M3 | §5.2 | **h_c is not fully specified.** The order of the window maximum and the county mean is open. "Customer-weighted" can only mean the population weights of the 3 km nodes. The gust field is also unspecified. | "Each ratio takes the population-weighted county-mean series of the ERA5 field over the 3 km nodes, then its maximum over the forecast window. The gust field is fg10." |
| M4 | §3.7, §10 | **S-b's inputs need pinning.** The code counts every row, zero and blank rows included; pin that. A national threshold of 5 counties detects only total collection failure, not partial failure (a stated limit). Amendment 1(5) says the collection-run set is outcome-free, but only for S-b. | "§10: the national collection-run set (S-b, G4, G5, forecast mask) is not a read of any tranche." |
| M5 | §6 | **Blank rows may not be random.** Blank-count rows may coincide with large outages, when a feed fails under load, so the forecast mask risks being not-at-random. | Diagnostic on D: the share of blank-masked county-hours for h_c ≥ 1 against h_c < 1. |
| M6 | §1 | **Some exclusions are structural but not stated.** The claim population omits them: windows containing Dec 31 of 2018-2021 (24 of 216 h missing, so S-b fails; a New Year hole in winter's core season); HRRR-gap windows (S-c); Connecticut after 2025-05-29 (amendment 1); used systems (C only); operator-excluded county-events. | Add them to §1's "Outside it" list. |
| M7 | §7 | **Operator-exclusion sources have gaps.** A DOE-417 record exists because an outage was large; a load-shed row can be a damaged system shedding load, which is storm damage. DOE-417 summaries for 2024-2025 were not found (literature §3), so 2024-2025 sheds rest on the five named events. NERC EEA3 declarations (statistical review §5) are unused. A PSPS event without a county list has no fallback rule. The records' local times need conversion to UTC. | "Rows whose narrative attributes the shed to damage in the reporting utility's own system are flagged and kept. A PSPS event without a county list excludes the IOU's EIA-861 territory for its window. All times are converted to UTC. The 2024-2025 gap is a stated limit." |
| M8 | §4.6, §3.1 | **The audit axes do not match the data.** "SD of 3DEP elevation" is not computed anywhere; the existing descriptor is `relief_p95_p5`. Distance to the coast is not computed. Descriptors exist for 1,756 counties (2,409 with e3), not for every gated CONUS county; `fetch_geography_conus.py` is in progress. §3.1 lists none of these sources. | Name one descriptor and one source per axis in §3.1 before step 3. |
| M9 | §9.3 | **Stage 0 details are unset.** The bootstrap has no stated draws or seed. Non-inferiority has no stated statistic. The "worst regime" has no stated set. | "2,000 draws, seed 20260924 (as H2b); non-inferiority on the seed-averaged point estimate; worst regime over all five." |
| M10 | §5.4 | **Trimming is unspecified and differs by tranche.** The median's population is not stated, and the trimmed D and C estimands differ. | "The median is over county-event weights of the regime in the tranche; both trimmed and untrimmed values are reported per regime." |
| M11 | §9.1 | **"Largest system" and a group's origin are undefined.** | "Largest = most sampled county-events; a group's origin is its earliest origin." |
| M12 | §10 | **The power table rests on other assumptions.** Its effective sizes come from a Monte Carlo of equal-probability draws with five headline regimes. PPS, trimming, used-window removal and possibly only three headline regimes lower them. | Recompute the Kish design effect of w on C's sample, outcome-free, before the confirmatory registration. |
| M13 | §1 | **The loss matches only one estimand.** The Z_r-normalised loss matches skill against zero. Gain against the host weights regimes by 1/SSE_r(host) instead. | State it. |
| M14 | §2, §4.1, §3.2, §6 | **Loose ends in §2-§6.** A CONUS window is 365 MB (main 236 + extra 108 + gust 22, `downloads.jsonl`), not 236 MB. At 29 GiB free, §8's deletion after features is mandatory. §4.1 does not say whether P is the whole next release or a §4.4 draw. SQ.W (snow squall warnings, polygons from 2018) is not listed. EAGLE-I years other than 2022 are not checked against the v4 checksums. | One line each. |

## 4. Review suggestions neither adopted nor listed in §14

| Review | Suggestion | v1 | Action |
|---|---|---|---|
| stat. rev. 3; phys. §4 | membership from products issued before the hazard; first-issuance areas | initial polygons (amendment 1(3)); zones including extensions | S12 |
| stat. §4 | size stratification, or the sealed winter stratum skews small | not addressed | S5 |
| stat. §2 | ring π graded by distance (near 1 close by, 0.05-0.1 far away) | one 200 km ring, PPS on h_c | §14 line: h_c replaces distance as the size measure |
| stat. §3, rev. 7 | heat and flood-only systems as negative-control strata | heat not sampled; flood-only systems are full heavy-rain systems | §14 line |
| stat. §5, rev. 6 | 2015-2018 as a secondary frame | dropped | §14 line |
| stat. rev. 14 | the host must beat zero in each regime | at least three headline regimes; the rest are non-inferiority strata | §14 line |
| stat. §5 | NERC EEA3 declarations as a source | not used | M7 |
| stat. §4 | commit the draw code and seed before drawing | seeds registered, code not | S13, plus "draw.py is committed before it runs, and it runs once" |
| phys. §6.3 | frequency-weighted headline | regime-balanced headline, as stat. §2 asks | §14 line: the two reviews differ, and v1 follows stat. |
| phys. §3 | hazard vector per county-hour; compound = the same county within 24 h | per county-event; label mixes | S2 |
| phys. §4 | report damaging and quiet sides separately | h_c classes, never outcome subsets | §14 line |
| phys. §1 | separate the sub-regimes (derecho vs pulse storms, wet vs dry snow) | not stratified | §14 line |
| lit. 6.7 | in-data coverage check before the window | minimum coverage only | S4 |
| lit. 6.8 | HRRR results for 2019-2025 against earlier years | split at HRRR v4 (2020-12-02) | §14 line |

## 5. Checked and found sound

* **Pre-window service rule** (`panel.in_service_before`, with `service_rule="pre_window"` in `build_panel`). It reads only
  [t0 − 30 d, t0), counts positive records only, and sets observed = in service × national run. The forecast window can
  mask hours but never drop a county.
* **The 16-day rule matches the old read span.** Old 216 h panels read service status from t0 − 7 d to t1 + 7 d
  (≈ t0 + 16 d), so a new 9-day window overlaps that span only if it starts within 16 days. The same holds for the 7-day
  main-line windows (day − 2 d to day + 5 d, read ±7 d).
* **The draw as coded.** C is drawn first from systems that are not used, with seeds fixed before the frame. Systematic
  PPS with iterative certainty gives exact first-order π. D's π are conditional on C, and C is never supplemented.
  `draw.py` uses the S1-S3 sets for used windows and for blocking.
* **Tropical allocation.** [frame]: 53 eligible tropical systems. C = 20 needs at most 16 of them to be used, and D = 15
  then needs at most 18 blocked. The draw's log should report both counts.
* **Loss and estimand algebra.** Σ_r (1/R) SSE_r(f) / Z_r = 1 − Φ against zero, with the same weights in loss and metric.
* **Testing budget.** Fixed-sequence testing that stops at the first non-rejection controls the familywise error at α.
* **Operator exclusions.** They are decided from type, footprint and time, compiled before the gates, and applied to
  every tranche. Filtering DOE-417 on the alert criterion handles the mistyped February 2021 rows.
* **Coverage counts in §6** (24 states in 2018, 39 in 2022) match the coverage table. H1a's withdrawal and H2b's closure
  match PREREG_W2 amendment 8.
* **Amendment 1 resolves:**
  * tropical products joining their storm by ETN (the first build had 19 stray "tropical" systems, e.g. Tampa Bay's
    surge and TS warnings for Michael, ETN 1014, 32 h before the Michael system's onset);
  * exposure through the initial polygon, and EW through its UGC rows;
  * the meaning of the time fields;
  * S-d's dependence on the exclusions (steps 1 and 2 committed together).
* **Feasibility.**
  * The frame sources are small: VTEC CSV about 11 MB a year, SBW polygons 6-7 MB a year, UGC versions 15 MB. The HRRR
    listing is done (65,784 hours, 99.8% present).
  * ERA5 from ARCO took about 2.5 min per window for the main set in the logged runs, so all of D's weather is a few
    hours of fetching.
  * HRRR byte-range builds took about 1 min per window with six fields.
  * About 10.5k county-events fits the 40-minute run.
  * Stage 0 is 30 runs, about 7 h on three workers.
