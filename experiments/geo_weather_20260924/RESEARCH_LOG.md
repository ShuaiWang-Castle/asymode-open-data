# RESEARCH_LOG — one line per attempt (newest last)

Format: time | id | hypothesis | change | screen result (RMSE, vs base, county interval) | keep/discard | files

| time | id | hypothesis | change | result | decision | files |
|---|---|---|---|---|---|---|
| 09-24 23:40 | S0 | - | branch opened; program.md, DESIGN v0 | - | - | program.md, DESIGN.md |
| 09-24 23:45 | D1 | customers' local conditions differ from the cell's | data v3 nodes: 150 m pixels -> 16 county-wide strata (dz quartiles x canopy x wet), WorldPop weights, ERA5 orography | 2,409 counties, 16 nodes each; max dropped population share 0.15 (24-cell cap) | superseded by D3 (strata must not mix cells) | build_subgrid_nodes.py, data_provenance/nodes_e3.json |
| 09-24 23:50 | D2 | - | node weather: cell mixture per node, population-weighted county weather | [6122, 16, 216, 9] | kept for audits | build_node_weather.py |
| 09-25 00:00 | D4 | L1w': exposure weighting matters (REVIEW_formal 5) | data v3p: every round-2 input rebuilt with population instead of area weights (WorldPop 2020 inside county x cell, +1% area floor) | built; mean change: gust 0.23 m/s, t2m 0.26 C, pressure 268 Pa | screen S1 running | build_v3p.py, data_provenance/features_v3p.json |
| 09-25 00:05 | M0 | learned local mechanism scalars | v0 LocalMechanisms into the first damage layer | step 2.9 s vs 0.44 s (fold size, 4 threads); reviews: zero gradient at step 0, clamping, weak identification | discarded before screening | src/asymode/geo_mech.py |
| 09-25 00:15 | S3 | DESIGN v1 (both reviews) | exposure-integrated hazard: fixed D, 10 gated features x 4 modulators x 4 memories, competing-hazard entry, beta >= 0 on a slow clock; cell x dz-band nodes | code + tests (arm equals W+Cin bit for bit at step 0) | build | DESIGN.md, build_nodes_cs.py, build_eih.py |
| 09-25 00:20 | S1 | - | quick screen: W+Cin on e3r2 (base) and on v3p, folds 1-2, 900 steps, seed 0 | running | - | screen.py, evaluate_screen.py |
