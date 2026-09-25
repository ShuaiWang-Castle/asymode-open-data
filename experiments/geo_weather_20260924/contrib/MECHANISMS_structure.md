# Mechanism physics for a local geo-weather outage model

Contribution to `DESIGN.md` §2(b)–(c): the physical relations, ERA5-computable intensity measures, local
geographic modifiers, bounded parameters and state time constants for six outage mechanisms, plus the shared
downscaling operators they rely on.

Provenance: compiled from the public sources listed at the end. Every number is tagged:
*src* = taken from the cited source; *assume* = a modelling choice, to be tested and learned here.

**Notation.** Cell c = the ERA5 grid cell; node k = a customer-weighted stratum inside it, with elevation z_k and
terrain, canopy and soil attributes. ERA5 short names: `t2m`, `d2m` (K); `tp`, `sf`, `cp` (m, accumulated over
the hour); `fg10` (maximum 10 m gust over the past hour, m/s); `u10`, `v10`; `swvl1`–`swvl2` (m³/m³, layers
0–7 and 7–28 cm); `cape` (J/kg); `sp` (Pa); `z` (surface geopotential; z_c = z/g is the *model* orography);
`t` at 850 hPa (pressure levels). σ(x) = 1/(1 + e^(−x)).

---

## 0. Shared operators

### 0.1 Temperature lapse

```
T_k = T_c − Γ (z_k − z_c)          z_c = ERA5 model orography of the cell, not the true mean elevation
```

* ERA5 `t2m` is valid at the model's own surface height. The correction must start from the geopotential
  field; starting from the true cell-mean elevation adds a systematic offset Γ × (model − true orography) to
  every node.
* Γ: init 6.5 K/km (standard atmosphere), or the month's climatological value, bounds [0, 9.8] K/km *assume*
  (upper bound = dry adiabatic).
  * Monthly Northern-Hemisphere values used by MicroMet, from Kunkel (1989): Jan 4.4, Feb 5.9, Mar 7.1,
    Apr 7.8, May 8.1, Jun 8.2, Jul 8.1, Aug 8.1, Sep 7.7, Oct 6.8, Nov 5.5, Dec 4.7 K/km (Liston & Elder
    2006, Table 1) *src*. Winter lapse rates are the shallowest.
  * Observed near-surface lapse rates are often shallower than 6.5 K/km: annual means of 3.9–5.2 K/km on
    the windward side of the Cascades (Minder et al. 2010) *src*.
* Limitation: valley cold pools give negative near-surface lapse rates, which a single linear Γ ≥ 0 cannot
  represent. Well-mixed precipitating conditions weaken them but do not remove them.

### 0.2 Dew-point lapse

MicroMet (Liston & Elder 2006, after Kunkel 1989) lapses dew point linearly, using a monthly vapour-pressure
coefficient λ:

```
Td_k = Td_c − Γ_d (z_k − z_c),     Γ_d = λ c / b,    b = 17.502, c = 240.97 °C (Magnus constants, water)
λ (1/km): Jan .41  Feb .42  Mar .40  Apr .39  May .38  Jun .36  Jul .33  Aug .33  Sep .36  Oct .37  Nov .40  Dec .40
⇒ Γ_d ≈ 4.5 (Jul–Aug) to 5.8 (Feb) K/km
```

* The λ table and the Γ_d formula are *src* (Liston & Elder 2006); the Γ_d range is computed from them.
  Kunkel (1989) found the dew-point lapse approximately constant with height. These are climatological values
  for the mountainous western United States.
* Always clip Td_k ≤ T_k.
* Near saturation during precipitation, the humidity deficit is small at every node. An alternative that
  respects this holds relative humidity fixed across the cell and derives Td_k from T_k and RH_c. Learn a
  mixing weight between the two forms *assume*.
* Bounds on Γ_d: [1.8, 8] K/km *assume*. The lower end is the dew-point lapse of an unsaturated parcel that
  conserves its mixing ratio (≈ 1.8 K/km, a textbook value).

### 0.3 Wet-bulb temperature

* **Preferred:** solve the psychrometric equation at node pressure
  p_k = sp_c · exp(−g (z_k − z_c) / (R_d T̄)).
* **Closed form:** Stull (2011) gives Tw(T, RH) at standard sea-level pressure. It is valid for RH 5–99 % and
  T −20…50 °C (except cold and dry together), with errors of −1…+0.65 °C and mean absolute error < 0.3 °C
  *src*. It ignores pressure, so use it only for low nodes or as an initial guess.
* Cross-check: Kunkel (1989) used a region-wide lapse of 4.4 K/km for summer design wet-bulb temperatures *src*.

### 0.4 Precipitation phase from the wet-bulb temperature

```
f_snow,k = 1 / (1 + exp((Tw_k − T50) / s))
P_snow,k = f_snow,k · P_k        P_liq,k = (1 − f_snow,k) · P_k
```

* **Why wet-bulb.** Ding et al. (2014) found wet-bulb temperature a better phase discriminator than air
  temperature, with thresholds that rise with surface elevation and depend on relative humidity. Wang et al.
  (2019) found that wet-bulb partitioning corrects the snowfall underestimate of air-temperature thresholds in
  dry air *src*.
* **T50:** init 1.0 °C, bounds [−0.5, 2.5] °C.
  * Source: the air-temperature threshold at which rain and snow are equally frequent averages 1.0 °C across
    Northern-Hemisphere stations, with 95 % of stations in −0.4…2.4 °C (Jennings et al. 2018) *src*.
  * The wet-bulb threshold equals the air-temperature threshold at saturation and lies at or below it
    otherwise *assume*.
* **s:** init 0.7 °C, bounds [0.2, 1.5] °C.
  * Source: Dai (2008) fits snow frequency over land with a hyperbolic tangent of air temperature, slope
    b ≈ 0.7 per °C, so the matching logistic scale is s = 1/(2b) ≈ 0.7 °C. The transition spans about
    −2…+4 °C over low-elevation land *src*.
  * Wet-bulb transitions are sharper *assume*.
* ERA5 snowfall (`sf`) is ERA5's own cell-level partition. It cannot move with node elevation, so use it only
  as a consistency check at z_k = z_c.
* **The elevation–phase chain.** At fixed cell weather, a node Δz above the model orography is about Γ·Δz
  colder, and its wet-bulb temperature follows. Its phase, its wet-snow window (§1) and its freezing-rain
  indicator (§2) all shift with Δz. The altitude band where each applies moves with every event's freezing
  level. This is why mechanisms 1–2 must be evaluated per node rather than on the cell mean.

### 0.5 Wind on exposed terrain

MicroMet (Liston & Elder 2006):

```
W_k = W_c · (1 + γ_s Ω_s,k + γ_c Ω_c,k)
Ω_s = terrain slope in the wind direction; Ω_c = curvature at a length scale ≈ half the ridge-to-valley
      wavelength; each rescaled to [−0.5, 0.5] over the domain
γ_s, γ_c ∈ [0, 1], suggested γ_s + γ_c = 1  ⇒  multiplier in [0.5, 1.5]
```

* Every element of this form is *src*: the rescaling, the weights and the suggested constraint.
* Physical basis: flow accelerates over hill crests roughly in proportion to the hill's slope H/L (linear
  theory, Jackson & Hunt 1975). Taylor & Lee (1984) give guidelines by feature type (ridges, escarpments,
  3-D hills).
* The slope term depends on wind direction, so it needs `u10` and `v10`.
* `DESIGN.md`'s W_k = W_c exp(κ TPI_k) is compatible if κ is bounded so that the multiplier stays within
  [0.5, 1.5] over the domain's TPI range *assume*.
* Apply the same multiplier to `fg10` and to the mean wind *assume*. Gust factors also change over terrain.

### 0.6 Local wind climatology

Trees and lines are adapted to their local wind climate. Klawa & Ulbrich (2003) model storm loss as increasing
with the cube of the gust's excess over the local 98th percentile:

```
loss ∝ Σ_t [ G_t / G98 − 1 ]_+^3
```

This form is *src*. Compute G98 per cell from a multi-year ERA5 `fg10` record, then carry it to each node with
the terrain multiplier of §0.5. The normalization needs no outage data. It removes much of the between-location
offset in damage thresholds without any location-specific parameter.

---

## 1. Wet-snow accretion (conductors and tree crowns)

**Physics.** Partially melted snowflakes stick to conductors and crowns. The sleeve or crown load grows until
it melts, sheds or breaks the structure. The general accretion rate on an object (Makkonen 2000; ISO 12494) is

```
dM/dt = α1 α2 α3 · w · A · v
```

* w = mass concentration of snow in air; A = cross-section; v = impact speed.
* α1 = collision efficiency (≈ 1 for snowflakes); α2 = sticking efficiency; α3 = accretion efficiency (≈ 1 for
  wet snow).

With snowfall rate P_s, flake fall speed v_f and wind U, w = P_s / v_f and v = √(U² + v_f²). Per unit length of
a cylinder of diameter D:

```
dM/dt = β(U) · P_s · √(U² + v_f²) / v_f · D
```

* **Sticking.** Nygaard et al. (2013) tested the commonly used inverse-wind sticking efficiency β = 1/U
  against 50 years of observed events. They found it strongly underestimates accretion. They also identify wet
  snow by a flake liquid-fraction criterion and revise the fall speed used for wet flakes *src*.
* **Window.** Wet snow accretes at air temperatures between 0 and +3 °C (ISO 12494, table of icing
  types) *src*.

**Intensity measure (per node, per hour).**

```
I_ws,k = P_snow,k · σ((T_k − T_lo)/w_T) · σ((T_hi − T_k)/w_T) · β(U_k) · √(U_k² + v_f²)/v_f
β(U)   = min(1, (U/U0)^(−q))
```

The wet-bulb snow fraction closes the window from above: as Tw rises the flakes turn to rain. The lower
σ-term closes it from below: below freezing the snow is dry and does not stick.

| parameter | init | bounds | basis |
|---|---|---|---|
| T_lo | 0 °C | [−1, 1] °C | ISO 12494 lower edge *src*; bounds *assume* |
| T_hi | 3 °C | [2, 4] °C | ISO 12494 upper edge *src*; bounds *assume* |
| w_T | 0.3 °C | [0.1, 1] °C | *assume* |
| v_f | 1.5 m/s | [0.5, 3] m/s | *assume* |
| U0 | 1 m/s | fixed | *assume* |
| q | 0.5 | [0, 1] | q = 1 is β = 1/U; q = 0 is full sticking. Nygaard et al. (2013): 1/U too weak *src* (direction); value *assume* |

**Geography.**
* **Elevation.** Acts through T_k, Td_k and Tw_k (§0.1–0.4). It is the dominant local modifier: the window
  occupies a narrow altitude band that follows the freezing level.
* **Canopy.** Tree crowns intercept wet snow. Snow loading of trees causes mainly stem breakage, and also
  bending and uprooting (Nykänen et al. 1997) *src*. Lines under or beside canopy fail from falling limbs.
  Weight the crown pathway by the node's forest fraction. Deciduous crowns intercept far more in leaf-on season
  (§3).
* **Exposure.** U_k (§0.5) enters the flux directly.

**State and shedding.**

```
dL_k/dt = I_ws,k − L_k / τ_ws(T_k, U_k)
```

* τ_ws is short when T_k is well above the window (melt). It is long when T_k drops below 0 °C after
  accretion: the sleeve freezes and persists. Bounds [1 h, 72 h] *assume*.
* Snow intercepted by canopies unloads roughly exponentially in time (Hedstrom & Pomeroy 1998). The unloading
  coefficients reported across implementations differ by an order of magnitude, so learn τ within the bounds.

---

## 2. Freezing rain and glaze

**Physics.** Supercooled rain freezes on contact with sub-freezing surfaces. Jones (1998) gives a simple,
conservative model: uniform radial ice on a horizontal cylinder, independent of its diameter.

```
R_eq = (1/(ρ_i π)) · Σ_{j=1..N} [ (P_j ρ_o)² + (3.6 V_j W_j)² ]^(1/2)      [mm]
W_j  = 0.067 · P_j^0.846   (g/m³, Best's formula)
P in mm/h, V in m/s, ρ_i = 0.9 g/cm³, ρ_o = 1.0 g/cm³
```

* Assumptions: collision efficiency 1, all impinging water freezes (conservative), uniform accretion. For
  example, 10 mm of rain with no wind gives 3.5 mm of radial ice (Jones 1998) *src*.

**Intensity measure.** Freezing rain needs a melting layer aloft over a sub-freezing surface layer, which 2 m
variables alone cannot see.

```
FZ_k   = σ((T_fz − T_k)/w_fz) · 1[warm nose]        warm nose: t(850 hPa) > 0 °C, or ERA5's diagnosed precipitation type
I_zr,k = FZ_k · (1 − f_snow,k) · [ (P_k ρ_o)² + (3.6 U_k W(P_k))² ]^(1/2) / (ρ_i π)
```

* T_fz: init 0 °C, bounds [−2, 0.5] °C *assume*. Surfaces cool radiatively, so glaze can form in air slightly
  above freezing. w_fz: init 0.3 °C *assume*.
* ERA5 provides a diagnosed precipitation-type field. Treat it as a noisy indicator: diagnosing freezing rain
  from model profiles is uncertain.
* The gate uses the node's liquid fraction (§0.4). Under the same cell weather, a lower node can be glazing
  while a higher node is snowing.

**Geography.**
* **Elevation and aspect.** Ice accumulation varies strongly with both. In one documented case it was
  24.4 mm at 606 m and 29.4 mm at 871 m (Irland 2000) *src*.
* **Wind after the ice.** Damage severity is closely related to the winds that follow the heaviest
  accumulation (Irland 2000) *src*, so keep a glaze × wind interaction with §3.
* **Glaze on snow.** Glaze immediately after heavy snowfall is highly damaging to vegetation (Irland 2000)
  *src*: add the wet-snow load of §1 to the vegetation load.
* **Species and stand structure.** They modulate damage strongly (Irland 2000). Use the canopy attributes in
  the fragility.

**State and melt.**

```
dL_ice,k/dt = I_zr,k − k_m · max(0, T_k − 0 °C)          (degree-hour melt; ice persists while T ≤ 0 °C)
```

* k_m bounds [0.05, 1] mm/(K·h) *assume*. Sublimation is neglected.

**Fragility anchor.**
* The US National Weather Service defines an ice storm as structural damage or at least 6.35 mm of ice
  accumulation (as cited by Irland 2000) *src*.
* A forest icing experiment applied 6.4, 12.7 and 19.0 mm of radial ice. The 50-year-return radial ice for
  New England is 19–32 mm (Campbell et al. 2020) *src*.
* Fragility median: init 12 mm, bounds [6, 30] mm *assume*, anchored on those values.

---

## 3. Wind loading and tree fall; leaf-on versus leaf-off

**Physics.** Drag on a crown is F = ½ ρ C_d A U². The turning moment at the stem base or root plate drives stem
breakage or uprooting (review: Gardiner et al. 2016). Foliage and crowns reconfigure in strong wind:
* leaves and leaf clusters reconfigure so that their drag coefficients fall with wind speed (Vogel 1989) *src*;
* whole crowns streamline too: at 20 m/s, crown frontal area fell to 20–37 % of its still-air value in black
  cottonwood, red alder and paper birch (Vollsinger et al. 2005) *src*.

Drag therefore grows as U^(2+b), with b ≤ 0.

**Intensity measure.**

```
x_k    = G_k / G98_k                    G_k = node gust (§0.5); G98_k = local 98th percentile (§0.6)
I_w,k  = [x_k − 1]_+^3                  (Klawa & Ulbrich 2003)
   alt:  I_w,k = [G_k^(2+b) − (c_thr G98_k)^(2+b)]_+      (drag-based)
```

| parameter | init | bounds | basis |
|---|---|---|---|
| exponent in the Klawa–Ulbrich form | 3 | fixed or [2, 4] | *src*; bounds *assume* |
| b (reconfiguration) | −0.5 | [−1, 0] | 0 = rigid bluff body, −1 = strongly reconfiguring *assume* |
| c_thr | 1.0 | [0.8, 1.3] | *assume* |

`DESIGN.md`'s quadratic excess [W − θ]_+² is the b = 0 case without local normalization. Normalizing by the local
climatology (§0.6) is recommended: it replaces a location-specific threshold with a physical one.

**Geography: exposure, canopy and leaf state.**

```
canopy factor = 1 + c_can · forest_k · (f_evergreen,k + m_leaf · f_deciduous,k)
m_leaf = 1 in leaf-on season;  m_leaf = m_off ∈ [0.2, 1], init 0.5, in leaf-off season   *assume*
```

* Leaf state comes from the calendar and phenology, or from ERA5's leaf area index of high vegetation
  (`lai_hv`, a monthly climatology) *assume*.
* Deciduous leaf-off removes most of the leafy crown area that the streamlining measurements above refer to.
* Forest composition: NLCD deciduous, evergreen and mixed forest classes.
* **Compound loading.** A crown carrying wet snow (§1) or glaze (§2) presents more area and mass. Add an
  interaction I_w × (L_ws + L_ice).

**State.** None for the loading itself. Cumulative exposure within an event is the running sum of I_w, which is
the Klawa–Ulbrich event sum.

---

## 4. Windthrow on saturated or shallow soils

**Physics.** Trees uproot when the wind's turning moment exceeds the anchorage of the root–soil plate.
* **Soil type and rooting depth.** Nicoll et al. (2006) analysed about 2000 conifers pulled over on 34 UK
  sites. Critical turning moment per unit stem mass differed among freely-draining mineral, gleyed (poorly
  drained) mineral, peaty mineral and deep-peat soils. Sitka spruce anchorage was poorest on gleyed mineral
  soils, and deep rooting (> 80 cm) raised critical turning moment by 10–15 % *src*.
* **Water below the root plate.** In tree-pulling tests with water supplied to the soil, high water content
  below the root plate decreased anchorage. Water inside the root plate first increased it, but reduced the
  hinge-side root-plate area (Kamimura et al. 2012) *src*.

**Intensity measure.**

```
I_wt,k = I_w,k · (1 + a_sat · S_k · d_k)
S_k = saturation state in [0, 1]
d_k = soil vulnerability of the node in [0, 1]: poorly drained, high water table, shallow to a restrictive layer
```

a_sat: init 0.3, bounds [0, 1] *assume*. The only anchorage magnitude verified above is the 10–15 % depth effect,
so the event-time saturation effect must be learned.

**Saturation state S (two options).**
1. **ERA5 soil water.** `swvl1`–`swvl2` scaled between the soil type's wilting point and saturation. This is a
   cell-level quantity for the top 28 cm.
2. **Antecedent precipitation index** (Kohler & Linsley 1951):

   ```
   API_t = k · API_(t−1) + P_t
   ```

   Values of k between about 0.80 and 0.98 per day are used in practice *src*. The hourly factor is
   k_h = k^(1/24); the e-folding time τ = −1/ln k runs from ≈ 4.5 d (k = 0.80) to ≈ 50 d (k = 0.98).
   `DESIGN.md`'s dS/dt = P − S/τ_S is the continuous form.

**Geography.**
* Poorer drainage lengthens τ_S. For example τ_S,k = τ_0 · exp(δ · poor_drain_k), with δ ≥ 0 and τ_S kept within
  [4.5 d, 50 d] *assume*.
* Shallow soils and a high water table raise d_k. A high topographic wetness index (§6) raises the local
  wetness.

---

## 5. Convective downbursts

**Physics.** Evaporative cooling and precipitation loading drive downdrafts whose outflow spreads along the
ground. Microbursts have an outflow extent under 4 km; macrobursts are larger (Fujita 1985). Their footprints
are far below ERA5 resolution.

**Intensity measure.**
* **Already in the gust.** ERA5's gust includes a parameterized convective contribution driven by deep
  convection and low-level wind shear (wind speed at 850 hPa minus 950 hPa) (Bechtold & Bidlot 2009). Part of
  the convective hazard therefore already enters mechanism 3.
* **Sub-grid hits.** Represent the part that is not in the gust as a hit probability:

  ```
  C_c      = σ((cape_c − CAPE0)/s_C) · cp_c/(cp_c + P0)          convective activity in the cell
  p_hit,c  = 1 − exp(−λ_d · C_c · Δt)                              expected fraction of the cell's area hit this hour
  I_db,k   = p_hit,c · E[ (G_burst / G98_k − 1)_+^3 ]
  ```

  CAPE0: bounds [100, 1000] J/kg *assume*. λ_d, P0 and s_C are *assume*.
* **Classical index.** WINDEX (McCann 1994) is the classical microburst-potential index. It needs the lapse
  rate and mixing ratio up to the melting level, so it requires pressure-level data.

**Geography.** There is little systematic terrain control at this scale. Exposure and canopy act as in §3 on the
nodes that are hit. Treat the hits as spatially random within the cell: the model predicts the expected damage,
not the location.

**State.** None (minutes).

---

## 6. Heavy rain and flooding

**Physics.** Heavy rain causes outages mainly in two ways:
1. flooding of substations and underground equipment;
2. soil saturation that feeds windthrow (§4).

Local terrain decides where water collects.

**Intensity measures.**
* **Runoff.** SCS curve number (USDA-SCS 1986, TR-55):

  ```
  Q = (P − 0.2 S)² / (P + 0.8 S)   for P > 0.2 S
  S = 25400/CN − 254               (mm)
  ```

  CN comes from soil hydrologic group × land cover and is adjusted for antecedent wetness.
* **Short-duration intensity.** Rolling 1-, 3- and 6-hour totals relative to local climatological quantiles,
  using the same exceedance logic as §0.6 *assume*.
* **Accumulation.** The event total and the API of §4.

**Geography.**
* **Topographic wetness index** TWI = ln(a / tan β) (Beven & Kirkby 1979): a = upslope area per unit contour
  width, β = local slope.
* **Height above the nearest drainage** (HAND; Nobre et al. 2011). It correlates with water-table depth.
* Node flood susceptibility: σ(−(HAND_k − h0)/s_h), with h0 and s_h *assume*.

**State.** Runoff routing takes hours to days; the API of §4 carries the slow part.

---

## 7. From node intensities to the county

* **Evaluate first, weight after.** Evaluate each mechanism on node weather and node geography, apply the
  fragility, and only then weight by customers:
  Λ_i,m = Σ_k ρ_k F_m(I_m,k).
  Averaging the weather first destroys the altitude-band structure of mechanisms 1–2, where the relevant band
  may hold only part of a cell's customers.
* **Fragility form.** Lognormal fragility in the load is standard for power-system components under wind
  (Panteli et al. 2017): F(I) = Φ(ln(I/I50)/σ_F), σ_F bounds [0.2, 1.0] *assume*. `DESIGN.md`'s softplus of a
  log-intensity ratio is a smooth monotone substitute.
* **Interactions to keep.**
  * glaze × wind (§2 × §3);
  * wet snow × wind (§1 × §3);
  * saturation × wind (§4).

## 8. Parameter summary

| operator / mechanism | parameter | init | bounds | basis |
|---|---|---|---|---|
| 0.1 T lapse | Γ | 6.5 K/km (or monthly) | [0, 9.8] K/km | Kunkel 1989 via Liston & Elder 2006; Minder et al. 2010 |
| 0.2 Td lapse | Γ_d | monthly, 4.5–5.8 K/km | [1.8, 8] K/km | Liston & Elder 2006 *src*; bounds *assume* |
| 0.4 phase | T50 (wet-bulb) | 1.0 °C | [−0.5, 2.5] °C | Jennings et al. 2018 |
| 0.4 phase | s | 0.7 °C | [0.2, 1.5] °C | Dai 2008 |
| 0.5 exposure | γ_s, γ_c | 0.5, 0.5 | [0, 1], sum ≤ 1 | Liston & Elder 2006 |
| 1 wet snow | T_lo, T_hi | 0, 3 °C | [−1, 1], [2, 4] °C | ISO 12494 |
| 1 wet snow | q (sticking) | 0.5 | [0, 1] | Nygaard et al. 2013 (direction) |
| 1 wet snow | τ_ws | 12 h | [1, 72] h | *assume* |
| 2 glaze | ρ_i, W(P) | 0.9 g/cm³, 0.067 P^0.846 | fixed | Jones 1998 |
| 2 glaze | T_fz | 0 °C | [−2, 0.5] °C | *assume* |
| 2 glaze | fragility median | 12 mm | [6, 30] mm | Irland 2000; Campbell et al. 2020 |
| 2 glaze | k_m | 0.2 mm/(K·h) | [0.05, 1] | *assume* |
| 3 wind | exponent | 3 | [2, 4] | Klawa & Ulbrich 2003 |
| 3 wind | b | −0.5 | [−1, 0] | Vogel 1989; Vollsinger et al. 2005 (direction) |
| 3 wind | m_off | 0.5 | [0.2, 1] | *assume* |
| 4 windthrow | k (API, per day) | 0.9 | [0.80, 0.98] | Kohler & Linsley 1951 |
| 4 windthrow | a_sat | 0.3 | [0, 1] | Nicoll et al. 2006; Kamimura et al. 2012 (direction) |
| 5 downburst | CAPE0 | 300 J/kg | [100, 1000] J/kg | *assume* |
| 7 fragility | σ_F | 0.5 | [0.2, 1.0] | Panteli et al. 2017 (form) |

## 9. What this specification cannot represent

* Temperature inversions and cold pools (§0.1).
* The placement of convective cells inside an ERA5 cell (§5).
* Known biases of reanalysis gusts, and the uncertainty of diagnosed freezing rain (§2).
* Vegetation management along lines, asset age and design standards, and repair logistics. All of these vary
  in space and are absent from the geography layer.

---

## References

* Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable contributing area model of basin
  hydrology. *Hydrological Sciences Bulletin*, 24(1), 43–69. doi:10.1080/02626667909491834
* Bechtold, P., & Bidlot, J.-R. (2009). Parametrization of convective gusts. *ECMWF Newsletter*, 119, 15–18.
  doi:10.21957/kfr42kfp8c
* Campbell, J. L., Rustad, L. E., Driscoll, C. T., et al. (2020). Simulating impacts of ice storms on forest
  ecosystems. *Journal of Visualized Experiments*, 160, e61492.
* Dai, A. (2008). Temperature and pressure dependence of the rain–snow phase transition over land and ocean.
  *Geophysical Research Letters*, 35, L12802. doi:10.1029/2008GL033295
* Ding, B., et al. (2014). The dependence of precipitation types on surface elevation and
  meteorological conditions and its parameterization. *Journal of Hydrology*, 513, 154–163.
* Fujita, T. T. (1985). *The Downburst: Microburst and Macroburst*. SMRP Research Paper 210, University of
  Chicago.
* Gardiner, B., Berry, P., & Moulia, B. (2016). Review: Wind impacts on plant growth, mechanics and damage.
  *Plant Science*, 245, 94–118. doi:10.1016/j.plantsci.2016.01.006
* Hedstrom, N. R., & Pomeroy, J. W. (1998). Measurements and modelling of snow interception in the boreal
  forest. *Hydrological Processes*, 12, 1611–1625.
* Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological
  Society*, 146(730), 1999–2049. doi:10.1002/qj.3803
* Irland, L. C. (2000). Ice storms and forest impacts. *Science of the Total Environment*, 262, 231–242.
  doi:10.1016/S0048-9697(00)00525-8
* ISO 12494:2017. *Atmospheric icing of structures*. International Organization for Standardization.
* Jackson, P. S., & Hunt, J. C. R. (1975). Turbulent wind flow over a low hill. *Quarterly Journal of the Royal
  Meteorological Society*, 101, 929–955. doi:10.1002/qj.49710143015
* Jennings, K. S., Winchell, T. S., Livneh, B., & Molotch, N. P. (2018). Spatial variation of the rain–snow
  temperature threshold across the Northern Hemisphere. *Nature Communications*, 9, 1148.
  doi:10.1038/s41467-018-03629-7
* Jones, K. F. (1998). A simple model for freezing rain ice loads. *Atmospheric Research*, 46, 87–97.
  doi:10.1016/S0169-8095(97)00053-7
* Kamimura, K., Kitagawa, K., Saito, S., & Mizunaga, H. (2012). Root anchorage of hinoki (*Chamaecyparis
  obtusa*) under the combined loading of wind and rapidly supplied water on soil: analyses based on
  tree-pulling experiments. *European Journal of Forest Research*, 131(1), 219–227.
* Klawa, M., & Ulbrich, U. (2003). A model for the estimation of storm losses and the identification of severe
  winter storms in Germany. *Natural Hazards and Earth System Sciences*, 3, 725–732.
  doi:10.5194/nhess-3-725-2003
* Kohler, M. A., & Linsley, R. K. (1951). *Predicting the Runoff from Storm Rainfall*. Research Paper 34,
  U.S. Weather Bureau.
* Kunkel, K. E. (1989). Simple procedures for extrapolation of humidity variables in the mountainous western
  United States. *Journal of Climate*, 2(7), 656–670.
  doi:10.1175/1520-0442(1989)002<0656:SPFEOH>2.0.CO;2
* Liston, G. E., & Elder, K. (2006). A meteorological distribution system for high-resolution terrestrial
  modeling (MicroMet). *Journal of Hydrometeorology*, 7(2), 217–234. doi:10.1175/JHM486.1
* Makkonen, L. (2000). Models for the growth of rime, glaze, icicles and wet snow on structures.
  *Philosophical Transactions of the Royal Society A*, 358, 2913–2939. doi:10.1098/rsta.2000.0690
* McCann, D. W. (1994). WINDEX—A new index for forecasting microburst potential. *Weather and Forecasting*,
  9, 532–541. doi:10.1175/1520-0434(1994)009<0532:WNIFFM>2.0.CO;2
* Minder, J. R., Mote, P. W., & Lundquist, J. D. (2010). Surface temperature lapse rates over complex terrain:
  Lessons from the Cascade Mountains. *Journal of Geophysical Research*, 115, D14122.
  doi:10.1029/2009JD013493
* Nicoll, B. C., Gardiner, B. A., Rayner, B., & Peace, A. J. (2006). Anchorage of coniferous trees in relation
  to species, soil type, and rooting depth. *Canadian Journal of Forest Research*, 36, 1871–1883.
  doi:10.1139/x06-072
* Nobre, A. D., Cuartas, L. A., Hodnett, M., et al. (2011). Height Above the Nearest Drainage – a hydrologically
  relevant new terrain model. *Journal of Hydrology*, 404, 13–29. doi:10.1016/j.jhydrol.2011.03.051
* Nygaard, B., Ágústsson, H., Somfalvi-Tóth, K., & Ólafsson, H. (2013). Modeling wet snow accretion on power lines:
  Improvements to previous methods using 50 years of observations. *Journal of Applied Meteorology and
  Climatology*, 52(10). doi:10.1175/JAMC-D-12-0332.1
* Nykänen, M.-L., Peltola, H., Quine, C., Kellomäki, S., & Broadgate, M. (1997). Factors affecting snow damage
  of trees with particular reference to European conditions. *Silva Fennica*, 31(2), 193–213.
  doi:10.14214/sf.a8519
* Panteli, M., Pickering, C., Wilkinson, S., Dawson, R., & Mancarella, P. (2017). Power system resilience to
  extreme weather: fragility modeling, probabilistic impact assessment, and adaptation measures. *IEEE
  Transactions on Power Systems*, 32(5), 3747–3757. doi:10.1109/TPWRS.2016.2641463
* Stull, R. (2011). Wet-bulb temperature from relative humidity and air temperature. *Journal of Applied
  Meteorology and Climatology*, 50, 2267–2269. doi:10.1175/JAMC-D-11-0143.1
* Taylor, P. A., & Lee, R. J. (1984). Simple guidelines for estimating wind speed variations due to small-scale
  topographic features. *Climatological Bulletin*, 18(2), 3–32.
* USDA Soil Conservation Service (1986). *Urban Hydrology for Small Watersheds*. Technical Release 55 (TR-55).
* Vogel, S. (1989). Drag and reconfiguration of broad leaves in high winds. *Journal of Experimental Botany*,
  40, 941–948.
* Vollsinger, S., Mitchell, S. J., Byrne, K. E., Novak, M. D., & Rudnicki, M. (2005). Wind tunnel measurements
  of crown streamlining and drag relationships for several hardwood species. *Canadian Journal of Forest
  Research*, 35(5), 1238–1249. doi:10.1139/x05-051
* Wang, Y., Broxton, P., Fang, Y., Behrangi, A., Barlage, M., Zeng, X., & Niu, G.-Y. (2019). A wet-bulb
  temperature-based rain-snow partitioning scheme improves snowpack prediction over the drier western United
  States. *Geophysical Research Letters*, 46(22), 13104–13113. doi:10.1029/2019GL085722
