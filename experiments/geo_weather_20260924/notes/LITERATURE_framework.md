# Literature map and adversarial review of DESIGN v0 (local geo-weather mechanisms)

Reviewer note, 2026-09-24. No model was trained for this note. Every citation was checked by web search in
this session (Appendix A). Section B3 quotes **ad hoc checks** run on the data v3 node files at the mechanism
layer's initial parameters. They are **not archived under `results/`** (program rule 5), and the snippet that
reproduces them is in Appendix B. Archive them as F0 before citing any of these numbers.

## 0. Verdict

1. The §1 identity is correct accounting under two hidden assumptions, and it is textbook: ecological bias,
   aggregation of nonlinear responses, disaggregation regression. It shows that county *means* are insufficient,
   not that county-level *information* is insufficient. It is an inductive-bias argument, not an impossibility
   result.
2. On the current panel and nodes there is little for it to act on. The 16-node quadrature is nearly collinear
   with one population-weighted node, and switching from area to population weighting moves the mechanisms
   roughly ten times more than the quadrature does. The panel is selected on wind reports and has no ice-storm
   event.
3. Flaw 1 is the interface. Outputs are standardised once, clamped, and read into the damage MLP by a
   zero-initialised matrix. The results are vanishing or noisy gradients, no monotonicity, and a static canopy
   offset in calm air.
4. Flaw 2 is the exposure and physics layer: county-wide strata that mix ERA5 cells, strata on axes that do not
   change the weather, a fixed lapse rate, freezing rain from surface temperature, people used as exposure, and
   served-pool depletion ignored.
5. Recommended: exposure-integrated hazards. Monotone, hazard-gated dictionary features at (cell × stratum)
   nodes enter as a competing hazard with non-negative coefficients. This nests W+Cin and has gradients linear
   in the coefficients. Run the cheap falsifiers (B4) first.

## Part A: literature map

### A1. Differentiable parameter learning and UDEs

* **Tsai et al. 2021**, *Nat. Commun.*, doi:10.1038/s41467-021-26107-z. dPL: a network maps attributes to
  process-model parameters and is trained through the differentiable model on many basins. Generalisation
  improves with data. This is the direct precedent for DESIGN (c)+(d).
* **Feng, Liu, Lawson & Shen 2022**, *WRR* 58:e2022WR032404, doi:10.1029/2022WR032404 (δ-HBV approaches LSTM
  accuracy with physical outputs). **Feng, Beck, Lawson & Shen 2023**, *HESS* 27:2357–2373,
  doi:10.5194/hess-27-2357-2023 (δ models degrade less than LSTMs in ungauged regions, the analogue of held-out
  counties).
* **Shen et al. 2023**, *Nat. Rev. Earth Environ.* 4:552–567, doi:10.1038/s43017-023-00450-9 (perspective).
* **Rackauckas et al. 2020**, arXiv:2001.04385 (UDEs). **Höge et al. 2022**, *HESS* 26:5085–5102,
  doi:10.5194/hess-26-5085-2022 (neural ODEs inside a hydrologic model structure).
* **Samaniego, Kumar & Attinger 2010**, *WRR* 46:W05523, doi:10.1029/2008WR007327 (MPR): transfer functions at
  fine scale, then upscaling operators, i.e. "local function, then integrate". **Feigl et al. 2022**, *WRR*
  58:e2022WR031966, doi:10.1029/2022WR031966, learns the transfer functions.
* **Bindas et al. 2024**, *WRR* 60:e2023WR035337, doi:10.1029/2023WR035337: reach-level parameters learned from
  attributes with supervision only at gauges, i.e. local parameters from aggregated observations.
* **Kratzert et al. 2019**, *HESS* 23:5089–5110, doi:10.5194/hess-23-5089-2019 (EA-LSTM: static attributes gate
  the network, structurally what GCRK did).
* **Heudorfer, Gupta & Loritz 2025**, *GRL* 52:e2024GL113036, doi:10.1029/2024GL113036: entity-aware gains come
  mostly from meteorology, and static attributes add little, even less out of sample. This is the hydrology
  version of RESULTS §18, finding 1.

*Lesson.* dPL works where each unit has many independent events and the forcing does not reveal the
attributes. Here each county has one to three 216-hour events, and ERA5 reveals the descriptors (RESULTS §18).
Sub-grid heterogeneity is absent from county-mean forcing, which is the right motivation for DESIGN, but its
size must be checked (B3).

### A2. Sub-grid heterogeneity and downscaling

* **Units, tiles, bands.** HRUs: Flügel 1995, *Hydrol. Process.* 9(3–4), doi:10.1002/hyp.3360090313. Mosaic
  tiles: Koster & Suarez 1992, *JGR* 97(D3):2697–2715, doi:10.1029/91JD01696. Aggregation review: Giorgi &
  Avissar 1997, *Rev. Geophys.* 35:413–437, doi:10.1029/97RG01754. Elevation classes: Leung & Ghan 1995,
  *Theor. Appl. Climatol.* 52:95–118, doi:10.1007/BF00865510. Topography-based subgrid structures: Tesfa & Leung
  2017, *GMD* 10:873–888, doi:10.5194/gmd-10-873-2017. VIC snow bands (model documentation, "Snow Bands
  Formulation"). Murillo et al. 2022, *WRR*, doi:10.1029/2022WR032113 (the sub-grid temperature distribution
  along bands matters for snow).
* **TopoSUB and TopoSCALE.** Fiddes & Gruber 2012, *GMD* 5:1245–1257, doi:10.5194/gmd-5-1245-2012, and 2014,
  *GMD* 7:387–405, doi:10.5194/gmd-7-387-2014. They cluster pixels on topographic attributes, force each cluster
  with reanalysis *interpolated from pressure levels to its elevation*, and aggregate with weights. This is
  DESIGN (a)+(b) almost verbatim, with a better downscaler.
* **Operators.** MicroMet: Liston & Elder 2006, *J. Hydrometeorol.* 7:217–234, doi:10.1175/JHM486.1 (source of
  the temperature lapse, dew-point lapse and terrain-wind operators). Gao, Bernhardt & Schulz 2012, *HESS*
  16:4661–4673, doi:10.5194/hess-16-4661-2012 (lapse rate from the reanalysis's own pressure levels). Minder,
  Mote & Lundquist 2010, *JGR* 115:D14122, doi:10.1029/2009JD013493 (surface lapse rates are often well below
  6.5 K/km). Bell & Bosart 1988, *MWR* 116:137–161 (cold-air damming, i.e. inversions).
* **Wind exposure.** Winstral, Elder & Davis 2002, *J. Hydrometeorol.* 3:524–538 (direction-dependent
  upwind-slope index Sx). Winstral, Jonas & Helbig 2017, *J. Hydrometeorol.* 18:335–348,
  doi:10.1175/JHM-D-16-0054.1.
* **Phase.** Jennings et al. 2018, *Nat. Commun.* 9:1148, doi:10.1038/s41467-018-03629-7 (the rain–snow
  threshold varies in space; humidity-aware methods do better). Harder & Pomeroy 2013, *Hydrol. Process.*
  27:1901–1914, doi:10.1002/hyp.9799.
* **Products.** ERA5: Hersbach et al. 2020, *QJRMS* 146:1999–2049, doi:10.1002/qj.3803; it includes an
  instantaneous precipitation-type field (ptype; ECMWF ERA5 documentation, Table 2). ERA5-Land at 0.1°:
  Muñoz-Sabater et al. 2021, *ESSD* 13:4349–4383, doi:10.5194/essd-13-4349-2021. HRRR at 3 km, hourly: Dowell et
  al. 2022, *Wea. Forecasting* 37(8), doi:10.1175/WAF-D-21-0151.1.
* **Unresolved weather.** Assumed-PDF closures (Golaz, Larson & Cotton 2002, *J. Atmos. Sci.* 59:3540–3551) are
  the template for convective gusts, whose placement inside a cell is random, not geographic.

### A3. Fragility, accretion, topology

* **Frameworks.** Ouyang & Dueñas-Osorio 2014, *Struct. Saf.* 48:15–24, doi:10.1016/j.strusafe.2014.01.001
  (hazard → fragility → network → restoration). Panteli et al. 2017, *IEEE TPWRS* 32(5):3747–3757,
  doi:10.1109/TPWRS.2016.2641463.
* **Empirical and component fragility.** Dunn et al. 2018, *Nat. Hazards Rev.* 19(1):04017019,
  doi:10.1061/(ASCE)NH.1527-6996.0000267: line fragility learned from a fault database, the closest analogue to
  learning F from outcomes. Salman, Li & Stewart 2015, *RESS* 144:319–333, doi:10.1016/j.ress.2015.07.028, and
  Shafieezadeh et al. 2014, *IEEE TPWRD* 29(1):131–139: poles and feeder series-system reliability.
* **Trees and ice.** Hou & Muraleetharan 2023, *IJDRS* 14(2):194–208, doi:10.1007/s13753-023-00478-x (tree
  failures). Hou et al. 2023, *RESS* 230:108964 (ice-storm fragility of poles and wires).
* **Multi-hazard.** Fu, Li & Li 2016, *Struct. Saf.* 58:1–10 (wind + rain). Mukherjee, Nateghi & Hastak 2018,
  *RESS* 175:283–305.
* **Accretion.** Jones 1998, *Atmos. Res.* 46:87–97, doi:10.1016/S0169-8095(97)00053-7 (ice load from rate and
  wind). Sanders & Barjenbruch 2016, *Wea. Forecasting* 31:1041–1060, doi:10.1175/WAF-D-15-0118.1 (FRAM).
  Makkonen 2000, *Phil. Trans. R. Soc. A* 358:2913–2939, doi:10.1098/rsta.2000.0690. Nygaard, Ágústsson &
  Somfalvi-Tóth 2013, *JAMC* 52(10), doi:10.1175/JAMC-D-12-0332.1 (wet snow identified by flake liquid
  fraction). Bonelli et al. 2011, *NHESS* 11:2419–2431, doi:10.5194/nhess-11-2419-2011. Canopy snow load with
  unloading: Hedstrom & Pomeroy 1998, *Hydrol. Process.* 12:1611–1625, the template for DESIGN's L-state.
* **Vegetation.** Radmer et al. 2002, *IEEE TPWRD* 17(4):1170–1175, doi:10.1109/TPWRD.2002.804006. Guikema,
  Davidson & Liu 2006, *IEEE TPWRD* 21(3):1549–1557, doi:10.1109/TPWRD.2005.860238.
* **Topology.** Winkler et al. 2010, *RESS* 95(4):323–336, doi:10.1016/j.ress.2009.11.002. Ji et al. 2016,
  *Nat. Energy* 1:16052, doi:10.1038/nenergy.2016.52: a small share of failures causes most customer
  interruptions, so impact is non-local and heavy-tailed.

### A4. Conditioning, identity memorisation, physics-informed ML

* **Conditioning.** FiLM: Perez et al. 2018, *AAAI* 32(1), doi:10.1609/aaai.v32i1.11671. HyperNetworks: Ha, Dai
  & Le, arXiv:1609.09106 (ICLR 2017). Mixture of experts: Jacobs et al. 1991, *Neural Comput.* 3(1):79–87, and
  Shazeer et al. 2017 (ICLR), which needs load-balancing losses against expert collapse. GCRK is a hypernetwork
  on static descriptors with little replication per unit, the textbook memorisation case. K8 is a hard-gated
  mixture of experts with few event × regime cells, and it failed too.
* **Failure modes.** Geirhos et al. 2020, *Nat. Mach. Intell.* 2:665–673, doi:10.1038/s42256-020-00257-z
  (shortcut learning). Roberts et al. 2017, *Ecography* 40:913–929, doi:10.1111/ecog.02881 (block CV). Essus,
  Vatsavai & Rachunok 2026, arXiv:2608.24665 (*preprint*): outage-model skill is inflated by autocorrelation, and
  leave-one-state or leave-one-event-out models often fail to beat a null.
* **PINNs and hybrids.** Raissi, Perdikaris & Karniadakis 2019, *J. Comput. Phys.* 378:686–707,
  doi:10.1016/j.jcp.2018.10.045. Karniadakis et al. 2021, *Nat. Rev. Phys.* 3:422–440,
  doi:10.1038/s42254-021-00314-5. Krishnapriyan et al. 2021, *NeurIPS* 34 (failure modes). Reichstein et al.
  2019, *Nature* 566:195–204, doi:10.1038/s41586-019-0912-1. Beucler et al. 2021, *PRL* 126:098302,
  doi:10.1103/PhysRevLett.126.098302 (hard architectural constraints).
  PINNs need a governing equation whose residual can be penalised, and outages have no field PDE. The trusted
  structure is a balance law (already hard-coded) plus empirical constitutive laws and downscaling operators.
  Use inductive bias (UDE/δ-model), not residual penalties, which bring PINN optimisation pathologies without a
  trustworthy residual.
* **Monotone classes.** I-splines with non-negative coefficients: Ramsay 1988, *Stat. Sci.* 3(4):425–441,
  doi:10.1214/ss/1177012761. Runje & Shankaranarayana 2023, *ICML* (PMLR 202).
* **Aggregate-output learning**, the statistical name for DESIGN §1 + §2d.
  * Ecological bias: Wakefield & Salway 2001, *JRSS-A* 164(1):119–137, doi:10.1111/1467-985X.00191.
  * Disaggregation regression: Law et al. 2018, *NeurIPS* (arXiv:1805.08463); Nandi et al. 2023,
    *J. Stat. Softw.* 106(11), doi:10.18637/jss.v106.i11; Arambepola et al. 2022, *Stat. Med.*,
    doi:10.1002/sim.9220 (recovery depends on the number and size of areas and on misspecification).
  * Functions of a distribution: DeepSets (Zaheer et al. 2017, *NeurIPS*); Szabó et al. 2016, *JMLR* 17(152).

### A5. Spatio-temporal outage models

* **Zhu, Yao, Xie, Qiu, Qiu & Wu 2025**, *INFORMS J. Data Sci.* 5(2):102–118, doi:10.1287/ijds.2023.0017
  (arXiv:2109.09711). A non-homogeneous Poisson model at town, zip or county level: weather enters through
  discounted exponential kernels and a DNN, and outages propagate through learned unit-pair weights. The
  weather memory is useful; the unit-pair weights can act as fingerprints.
* **Point processes and graphs.** Ertekin, Rudin & McCormick 2015, *Ann. Appl. Stat.* 9(1):122–144. Owerko,
  Gama & Ribeiro 2018, *IEEE GlobalSIP* 743–747, doi:10.1109/GLOBALSIP.2018.8646486 (GNN). Shen et al. 2026,
  arXiv:2604.04916 (*preprint*; contrastive location embeddings, i.e. unit-specific again).
* **Local-then-aggregate.** Wanik et al. 2015, *Nat. Hazards* 79(2):1359–1384, doi:10.1007/s11069-015-1908-2.
  Cerrai et al. 2019, *IEEE Access* 7:29639–29654, doi:10.1109/ACCESS.2019.2902558. Cerrai et al. 2020, *SEGN*
  21:100294, doi:10.1016/j.segan.2019.100294 (snow/ice models on a 4-km grid and by town; the top predictors are the
  amount of assets, LAI, snow density and, for ice, freezing rain). McRoberts, Quiring & Guikema 2018, *Risk Anal.* 38(12):2722–2737,
  doi:10.1111/risa.12728.
* **Data.** Brelsford et al. 2024, *Sci. Data* 11:271, doi:10.1038/s41597-024-03095-5 (EAGLE-I).

*Gap.* "Local, then aggregate" exists operationally, with asset maps as exposure and 2–4 km NWP as weather.
None of these works learns shared local mechanisms from county hourly trajectories through a two-rate
population balance. That is the defensible novelty.

## Part B: adversarial review

### B1. The identity

The identity assumes that failures at x interrupt customers at x and that the served pool is uniform in space.
Neither holds in a storm.

* **Depletion.** The host damages (1−p), so the effective county rate is u_eff = ∫ρ(1−p)λ / ∫ρ(1−p) ≠ Λ. Where λ
  is high, p is already high, so u_eff < Λ, and this saturation depends on the event's history.
  *Fix (exact, parameter-free):* node stocks p_k with the host recursion, recovery shared by the county, and
  p_i = Σρ_k p_k.
* **Topology.** A customer is out when any component on its feeder path fails (a series system: Salman 2015;
  Winkler 2010). So Λ_i = Σ_c h_c N_c, where N_c counts the customers downstream of component c. The measure
  lives on lines, not homes, and it is heavy-tailed (Ji 2016): ridge-crossing lines serve valley customers. The
  same tail puts compound-Poisson noise into the county × event residual, noise no mechanism can remove.
* **Overclaim.** The node set (ρ_k, g_k) *is* a county-level geography vector. The correct claim is that damage
  is a linear functional of the county's exposure measure under a shared local response, with *dynamic*
  sufficient statistics ∫ψ_j(w_t(g), g)dμ_i that no static descriptor reproduces. The minimal test therefore
  uses precomputed moments with a linear read-out, not an end-to-end mechanism network.
* **ICC 0.06 does not single out the Jensen gap.** The residual also contains compound noise, convective
  placement (sub-grid weather, not geography), restoration (falls hold about 27% of W's squared error in the
  five-event main design, RESULTS §11) and reporting error. The Jensen hypothesis predicts a structure: the
  residual grows with spread × proximity to curvature. Test that structure (F1).
* **Second order.** Λ − λ(z̄) ≈ ½Σ∂²λ·Cov_i(z). Only attributes that change the local weather, covary with it,
  or act nonlinearly create a gap. Canopy or soil multiplying a cell-constant hazard collapses to a weighted
  mean.
* **Example.** A county mean of +1 °C already lies near the rain–snow threshold (Jennings 2018), so it
  does not isolate the sub-grid effect. Use +3 °C with 30% of customers 500 m higher.

### B2. New versus known

**Known.**
* The identity (ecological bias, disaggregation regression).
* Quadrature plus downscaling (TopoSUB/TopoSCALE, bands, MicroMet).
* Parameter learning through a process model (dPL/δ, UDE, MPR).
* Fragility and accretion physics.
* Local-then-aggregate outage models.

**Plausibly new.**
* Dynamic disaggregation regression through a population-balance host.
* A diagnostic-and-placebo account of when static conditioning fingerprints and when sub-grid integration can
  inform.
* Node stocks over an exposure measure.

**Main novelty risk.** The gain may come from exposure weighting or a better weather product (B3.1). The work
then reads as data engineering, and reviewers will call the rest "TopoSUB + disaggregation regression +
fragility features". A fixed-physics feature baseline without learning is mandatory.

### B3. Failure modes, with evidence from the current code (ad hoc, initial parameters)

1. **Sub-grid weather is nearly constant.**
   * In `nodes_e3.npz`, node-mean TPI lies in about −24 to +7 m (1st–99th percentile), so exp(κ·TPI/100) is at
     most about 1.03 even at κ's bound, and mechanism 0's terrain term is inert.
   * The population-weighted spread of node dz within a county has a median of about 20 m; about one county in
     fifteen exceeds 100 m, which is about 0.75 K of spread at the 95th-percentile county.
   * The 16-node intensities correlate at least 0.98 with one population-weighted node. The quadrature explains
     about 3% of wet-snow variance and at most about 1% for the other mechanisms.
   * Area versus population weighting of that node leaves about 25% (wet snow) and 11% (snow/ice) of variance
     unexplained. The first-order effect is *where the customers are*.
2. **The panel lacks the mechanism's events.** Events were selected on wind-report footprints (RESULTS §10). In
   `event_selection.csv` no event lists Ice Storm among its top types, and six of twelve are warm-season
   (April–August) convective events. With about five effective events (RESULTS §15), pooled RMSE cannot certify a
   winter-phase mechanism.
3. **The code departs from DESIGN (a).** `build_subgrid_nodes.py` stratifies county-wide (dz quartiles ×
   canopy × wet), and `build_node_weather.py` mixes cells by population before the nonlinearities.
   * TPI is averaged away.
   * The dz tails are diluted into quartiles.
   * The cell-mixing loss that round 2 removed comes back (RESULTS §10).
   * Canopy and wet splits spend nodes on modulators that act linearly.
4. **The physics is wrong where it matters.**
   * A fixed Γ cannot represent inversions.
   * In `geo_mech.py`, liquid water needs T_w above `snow_t` (≈1 °C) while `r_fz` needs T < 0, and T_w ≤ T.
     So "freezing rain" is a few percent of precipitation near 0–1 °C, collinear with wet snow. Freezing rain
     needs a warm layer aloft.
   * The v0 wet-snow band was centred below 0 °C on dry-bulb temperature. That is colder than the partially melted
     regime (Makkonen; Nygaard) and colder than the 0–3 °C window in `contrib/MECHANISMS_structure.md` §1.
   * Use pressure-level profiles (TopoSCALE; Gao 2012) with ERA5 ptype, or HRRR.
5. **The interface has gradient pathologies** (`asym_host.attach_mechanisms`).
   * The zero-initialised M gives every mechanism parameter an exact zero gradient at step 0. Adam then scales
     the tiny, M-noise-driven gradients up to full-size steps, and parameters can drift to their sigmoid bounds,
     where gradients vanish.
   * The standardisation is frozen at initialisation, goes stale as parameters move, and makes the gains
     scale-degenerate with the rows of M.
   * `clamp(−10, 10)` has zero gradient outside its range. For wet snow about 6% of active county-hours are
     clamped, and they hold about 43% of the channel's intensity: the gradient is removed in the most intense
     hours.
   * The MLP read-in discards monotonicity and re-mixes the mechanisms.
   * The softplus floor makes wind loading non-zero in calm air, multiplied by (1 + a_can·canopy). That is a
     static county offset, and it breaks the hazard gating DESIGN §3 relies on.
   * A 900-step screen cannot move zero-started couplings (RESULTS §18, finding 4), so it tests *literature
     parameters as features*.
6. **Identifiability.**
   * Mechanisms co-occur on shared drivers (wind with wet snow, wind with saturation), so the county features are
     collinear.
   * A temperature bias and the band edges are exactly confounded; κ trades off against θ_w, and τ against gain.
   * One or two events inform each mechanism, so a learned mechanism can memorise *an event's* spatial field,
     which county-grouped folds cannot detect.
7. **Weights are not exposure.**
   * WorldPop counts people, not meters and not lines.
   * Undergrounding concentrates in dense, low areas, so the weight error correlates with elevation, i.e. with
     the gap being sought.
   * The 1% floor weights empty land.
8. **Evaluation.**
   * Learned-mechanism claims need an event-grouped check.
   * The false-positive rate of the seed-0, two-fold screen is uncharacterised (per-unit differences are weakly
     seed-reproducible, RESULTS §0).
   * Node sets borrowed from other counties create unrealistic dz spreads, so that placebo can fail for the wrong
     reason. Add within-county attribute shuffles, time-shifted weather and wind-direction rotation.

### B4. Minimal falsification sequence (each step can end the line)

* **F0, variance audit (no training).** With fixed literature parameters, compute features three ways:
  (a) area-weighted cell means, as the host uses now; (b) population-weighted; (c) per (cell × stratum)
  quadrature. Report the outage-weighted share of county-hours where (c)−(b) and (b)−(a) are material.
  *Kill* the quadrature if (c)−(b) is negligible there.
* **F1, residual structure (no training).** Regress the out-of-fold W+Cin residuals (county × event and hourly)
  on (c)−(b), on (b)−(a), and on spread × threshold-proximity terms, against a permutation null within
  event × state. Check whether the relative residual variance scales like 1/customers (compound noise).
  *Kill* if the gap features do not beat the null.
* **F2, fixed-feature screen** through Part C's coefficients. Arms: weights only, quadrature, and the B3.8
  placebos. Apply the two-numbers rule.
* **F3, planted recovery.** Plant with exact exposure, fit with proxy weights, and add compound noise.
* **F4, learnable parameters and node stocks,** only after F2 passes, with an event-grouped check.
* **Oracle downscaler.** HRRR or ERA5-Land features. If they do not beat population-weighted ERA5, lapse-rate
  downscaling will not either.

## Part C: a general formulation (exposure-integrated hazard)

1. **Nodes.** k = (ERA5 cell × stratum). Stratify only on axes that change the weather: dz relative to ERA5
   orography, with finer bins in the tails, and direction-aware exposure Sx(θ) at 1–5 km. Canopy along roads,
   drainage and leaf-on are within-node means. Exposure weights are ω ∝ pop^a·(road length)^(1−a), with a
   single learnable a ∈ [0, 1].
2. **Deterministic D.** Pressure-level profiles are interpolated to node elevation (TopoSCALE); the phase comes
   from ptype plus node wet-bulb; the exposure comes from Sx at the hour's wind direction. There is no learnable
   Γ.
3. **Dictionary.** Monotone I-spline ramps ψ_b at fixed knots of physical intensities (gust, Jones/FRAM ice,
   wet-snow accretion, interception load, saturation). Each is passed through memories
   τ ∈ {1, 3, 6, 12, 24, 48} h and multiplied by a modulator m_a(g) ≥ 0 that is zero wherever ψ_b is. Thresholds
   become knot weights.
4. **Features.** Φ_ij(t) = Σ_k ω_ik m_a(g_k) ψ_b(F_τ ∗ D(W_c(t), g_k)), precomputed once per data version.
5. **Host.** u = cap·[1 − (1 − u_W/cap)·exp(−Σ_j β_j Φ_ij)], with β_j = softplus(θ_j). This is a competing hazard
   (independent causes, the standard multi-hazard form) and equals the base at β = 0. At β = 0,
   ∂u/∂β_j = (cap − u_W)·Φ_ij, which is non-zero wherever the feature is: no slow start, no clamp, no stale scale.
6. **Diagnostics before training.** The effective rank of Φ, and the partial R² of Φ given the host inputs (the
   redundancy test of RESULTS §18).

**Properties.**
* Monotone and nested.
* Hazard-gated: the county-specific part varies with the weather by construction, which makes DESIGN §3 true
  rather than hoped.
* Linear in the exposure measure: a county exists only through its dynamic moments.
* Screens take minutes.

**Extensions, in order.**
* Low-rank coupling β_ab = Σ_r u_ar v_br ≥ 0: latent mechanisms, each a geography profile × weather response.
* Assumed-PDF sub-grid weather for convective gusts (half the panel).
* A monotone neural field φ(z_k, g_k) (Runje 2023) once the dictionary shows signal.
* Node stocks (B1).
* A frailty-depletion state for multi-wave events (Vaupel, Manton & Stallard 1979, *Demography* 16(3):439–454,
  doi:10.2307/2061224).

**Do first:** population-weighted host inputs (`pop_weather` already exists in `node_weather_e3.npz`). It is the
largest effect in B3.1; report it honestly as data engineering.

## Appendix A: verification

All works were confirmed by web search on 2026-09-24 (title, authors, venue; DOI where given).
* Preprints, verified to exist but not peer-reviewed: Essus et al. 2026; Shen et al. 2026.
* Documentation rather than papers: VIC snow bands; ERA5 ptype.
* Page ranges or DOIs I could not confirm are omitted, not guessed. No item is UNVERIFIED.

## Appendix B: ad hoc checks (reproduce, then archive under `results/`)

```python
# repo root, ./.venv/bin/python; inputs data/interim/geo_weather/{nodes_e3,node_weather_e3}.npz
import numpy as np, torch, sys; sys.path.insert(0, "src")
from asymode.geo_mech import LocalMechanisms, NODE_CH
Z = np.load("data/interim/geo_weather/node_weather_e3.npz"); ch = list(Z["channels"].astype(str))
nw = torch.from_numpy(Z["node_weather"][..., [ch.index(c) for c in NODE_CH]].astype(np.float32))
na = torch.from_numpy(Z["node_attr"].astype(np.float32)); nr = torch.from_numpy(Z["node_rho"].astype(np.float32))
m, S = LocalMechanisms(), range(0, len(nw), 500)
with torch.no_grad():
    lam16 = torch.cat([m(nw[s:s+500], na[s:s+500], nr[s:s+500])[:, 72:] for s in S])
    lam1 = torch.cat([m(torch.einsum("bk,bktc->btc", nr[s:s+500], nw[s:s+500])[:, None],
                        torch.einsum("bk,bka->ba", nr[s:s+500], na[s:s+500])[:, None],
                        torch.ones(len(nw[s:s+500]), 1))[:, 72:] for s in S])
z = (lam16 - lam16.mean((0, 1))) / (lam16.std((0, 1)) + 1e-6)     # as set_mechanism_scale
# per mechanism: corr(lam1, lam16); intensity share with |z| > 10 among lam16 > 1e-3.
# Area weighting: features_e3r2.npz area-mean channels with area_share-weighted node attributes.
# Node TPI and the rho-weighted sd of node dz: nodes_e3.npz attr.
```
