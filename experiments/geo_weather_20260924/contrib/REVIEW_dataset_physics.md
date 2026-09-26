# Physical review of DATASET_DESIGN v0

Provenance: **[generic]** — meteorology and outage physics from the public sources listed at the end.

## Summary

* **Type episodes by the parent weather system**, with a multi-label hazard vector per county-hour; warning
  types co-occur within one system.
* **Stratify geography inside each regime.** Stratifying by the geography that the regime's mechanism acts on
  (relief for winter, canopy for wind, drainage for rain) is what makes geography × weather interactions
  learnable; macro-regions do not guarantee it.
* **Leaf state is confounded with regime through season.** The panel needs shoulder-season episodes of the same
  regimes.
* **Heat and extreme cold are a different mechanism class** (thermal load, operator action). Defer them from v0
  or give them their own sub-system.
* **The dictionary lacks three things:**
  * temperature-gated load states (it only has linear filters);
  * a way to tell freezing rain from snow at the same surface wet-bulb temperature;
  * node-level compound products (load × wind, saturation × wind).

## 1. Regimes and the sub-regimes that differ physically

| Regime | Sub-regimes the panel must separate | Why they differ |
|---|---|---|
| Convective wind | organized MCS and derecho; QLCS / bow echo; supercell and tornado; pulse (air-mass) storms | Footprint and duration differ by orders of magnitude. A derecho's damage swath runs at least 650 km under the revised definition (Corfidi et al. 2016). Pulse storms give isolated, minutes-long wet microbursts that are sub-grid and stochastic. Deep-layer shear separates organized from pulse convection, and CAPE × precipitation alone cannot. |
| Synoptic wind | cold-season extratropical cyclones; terrain-forced downslope and gap winds; rarer warm-season synoptic wind | Extratropical storms are long-lasting and mostly leaf-off. In downslope winds the terrain produces the wind itself, not only the response to it. |
| Tropical cyclone | wind-dominated landfall; rain-dominated slow or inland remnants; coastal surge | Different mechanisms (structural and tree failure, inundation, salt water). Restoration lasts days to weeks. |
| Winter | wet snow; dry snow / blizzard; freezing rain (glaze); sleet; lake-effect snow; early- or late-season snow on leafy trees | Wet snow sticks at air temperatures of 0 to +3 °C (ISO 12494); dry snow accretes little. Glaze load follows Jones (1998), and ice-storm damage depends strongly on the wind after icing (Irland 2000). Sleet accretes little. |
| Heavy rain | flash flood (hours, convective); riverine flood (days, basin-scale); rain-on-snow | A riverine flood is driven by upstream rainfall, not local rainfall. |
| Heat, extreme cold | — | Transformer thermal overload and demand-driven operator action. Geography acts through load density and cooling demand, not vegetation. |

## 2. What geography × weather needs from the panel

An interaction g × w can be learned only if, within one regime, the panel spans the geography where the weather
is strong, and spans the weather across the geography. The draft's macro-region strata do not guarantee this.
Within each regime, stratify counties or episodes by the geography that the regime's mechanism acts on, with a
quota per stratum:

* **Winter × relief.** Freezing-level crossings need mountains and foothills (Appalachians, New England, Upper
  Midwest, Rocky Mountain front, Pacific ranges) next to flat terrain. Glaze needs open and forested landscapes
  under the same kind of storm (Southern Plains against Southeast).
* **Wind × canopy.** Forested East and South against the open Plains and West, and evergreen against deciduous.
  The coverage gate drops Nebraska, Kansas and North Dakota, which are the main canopy-poor high-wind states.
  Without some canopy-poor counties (included with coverage weights), the canopy modulator is identified only
  from variation within forests.
* **Season × canopy.** Convective storms happen mostly in leaf-on months and synoptic storms mostly in leaf-off
  months, so leaf state and regime are confounded. Add deliberate shoulder-season episodes: autumn convective
  lines, spring and autumn synoptic storms, early and late snow.
* **Wind × soil.** Include poorly drained counties (Gulf Coast, river bottoms) against well-drained ones, in wet
  and dry antecedent conditions. Anchorage is weakest on poorly drained soils (Nicoll et al. 2006) and when water
  sits below the root plate (Kamimura et al. 2012). In winter, soil frost changes the combined wind and snow risk
  (Gregow et al. 2011).
* **Structural confounders.** The urban–rural contrast (undergrounding share, vegetation-management cycles,
  utility practice) correlates with geography. Add a structural stratum or covariate so that geography is not
  credited with utility effects.

## 3. Typing mixed episodes

* **Stratify by parent system.**
  * Tropical cyclone: proximity to the best track in HURDAT2 (Landsea & Franklin 2013).
  * Extratropical cyclone: warnings clustered along one synoptic system.
  * Organized convection: an MCS.
  * Heat or cold: a persistent anomaly.

  A tropical storm's tornado and flash-flood warnings stay inside the tropical episode as components.
* **Record a hazard vector per county-hour.** Keep the warnings in effect together with the physical intensity
  measures. Analyse by component, stratify by parent system.
* **Compound episodes are their own sub-stratum.** Examples: ice then wind, or tropical rain then wind, in the
  same county within about 24 h. Their mechanisms are distinct (load × wind, saturation × wind) and should be
  sampled deliberately rather than by accident.
* **Independence is counted in parent systems.** Warnings on the same day from one synoptic system are one event
  for folds, power and the sealed set.

## 4. Warnings, windows and controls

* **Residual outcome dependence.** Warnings are issued before damage but react to incoming reports (extensions,
  re-issuance). Use first-issuance times and initial polygons. Criteria also vary by forecast office and changed
  over 2015–2025, so the period stratum is needed.
* **Windows.** Restoration time scales differ by regime: hours to days for convection, days to weeks for tropical
  cyclones and ice storms. A single 144 h window truncates long restorations and pads short ones with calm hours.
  Tie the window to warning end plus a regime-specific restoration tail.
* **Controls.** Random counties without warnings are mostly calm and teach little. Sample controls stratified by
  physical intensity instead, including near-misses with strong weather just below warning criteria, with known
  weights. Report errors on the damaging side and the quiet side separately.

## 5. Mechanisms the dictionary still misses

The current dictionary has gust ramps; liquid precipitation; precipitation × wet-bulb hats; CAPE × precipitation;
modulators for canopy, calendar leaf-on and poorly drained share; and linear memories of 3, 12 and 48 h. It misses:

1. **Local wind climatology.** The ramps are absolute, but damage scales with exceedance of the local 98th
   percentile (Klawa & Ulbrich 2003). `contrib/build_era5_fg10_g98.py` exists but is not in the dictionary.
2. **The warm layer aloft.** Freezing rain and snow can reach the surface at the same wet-bulb temperature, and
   the −1.5 °C hat mixes them. Add an elevated-warm-layer indicator (850 hPa temperature) or HRRR's categorical
   precipitation type (Dowell et al. 2022).
3. **Temperature-gated load states.** Glaze persists until a thaw; wet snow sheds on warming. Fixed-τ linear
   filters cannot represent either. Use accumulation while conditions hold and removal gated by temperature
   (Jones 1998; ISO 12494). These states are nonlinear, so they must be computed at the node before integration.
4. **Node-level compound products.** Load × subsequent gust and saturation × gust must be formed before the
   exposure integral: the county integral of a product is not the product of the integrals.
5. **Dynamic soil state.** A static drainage share does not know whether the soil is wet now. Add antecedent
   precipitation or soil water × drainage, and soil frost in winter.
6. **Phenological leaf state.** Green-up and leaf fall depend on latitude and elevation, which a May–October
   calendar ignores. This matters most for early and late snow and for autumn storms.
7. **Flood terrain and basins.** Add pooling terrain (HAND, Nobre et al. 2011; TWI, Beven & Kirkby 1979) and
   upstream basin rainfall for riverine floods; add coastal elevation for surge.
8. **Organization of convection.** Add deep-layer shear and the duration of strong gusts, to separate organized
   systems from pulse storms.
9. **Terrain wind exposure.** Slope in the wind direction and curvature (Liston & Elder 2006). Such effects are
   small at ERA5 scale, but HRRR's 3 km winds partly resolve exposure, so one HRRR-based test is justified.
10. **Heat or cold load.** If heat or cold stays in the panel, add consecutive-day temperature with a night-time
    minimum, times load density.

## 6. The draft's open questions

1. **Warnings.** They reduce outcome selection but do not remove it (§3–4).
2. **Controls.** Use intensity-stratified near-misses with sampling weights (§4).
3. **Allocation.** Equal allocation for learning and per-regime metrics; a frequency-weighted headline for
   operational relevance; report both.
4. **Power.** Count independent parent systems, not episodes. Six per regime is thin for any sub-regime claim;
   fewer regimes with more systems each (heat and cold deferred) buys power.
5. **Operator-initiated outages.**
   * DOE Form OE-417 public summaries list event type, demand loss and customers affected (threshold 50,000
     customers for one hour).
   * The California PUC publishes PSPS post-event reports.
   * Shutoffs below these thresholds will be missed.

## References

* Beven, K. J., & Kirkby, M. J. (1979). *Hydrological Sciences Bulletin*, 24(1), 43–69.
* Corfidi, S. F., Coniglio, M. C., Cohen, A. E., & Mead, C. M. (2016). A proposed revision to the definition of
  "derecho". *Bulletin of the American Meteorological Society*, 97(6), 935–949.
* Dowell, D. C., et al. (2022). The High-Resolution Rapid Refresh (HRRR), Part I. *Weather and Forecasting*,
  37(8). doi:10.1175/WAF-D-21-0151.1
* Gregow, H., Peltola, H., Laapas, M., Saku, S., & Venäläinen, A. (2011). Combined occurrence of wind, snow loading
  and soil frost with implications for risks to forestry in Finland. *Silva Fennica*, 45(1), 30.
  doi:10.14214/sf.30
* Irland, L. C. (2000). Ice storms and forest impacts. *Science of the Total Environment*, 262, 231–242.
* ISO 12494:2017. *Atmospheric icing of structures*.
* Jones, K. F. (1998). A simple model for freezing rain ice loads. *Atmospheric Research*, 46, 87–97.
* Kamimura, K., Kitagawa, K., Saito, S., & Mizunaga, H. (2012). *European Journal of Forest Research*, 131(1),
  219–227.
* Klawa, M., & Ulbrich, U. (2003). *Natural Hazards and Earth System Sciences*, 3, 725–732.
* Landsea, C. W., & Franklin, J. L. (2013). Atlantic hurricane database uncertainty and presentation of a new
  database format. *Monthly Weather Review*, 141(10), 3576–3592.
* Liston, G. E., & Elder, K. (2006). MicroMet. *Journal of Hydrometeorology*, 7(2), 217–234.
* Nicoll, B. C., Gardiner, B. A., Rayner, B., & Peace, A. J. (2006). *Canadian Journal of Forest Research*, 36,
  1871–1883.
* Nobre, A. D., et al. (2011). Height Above the Nearest Drainage. *Journal of Hydrology*, 404, 13–29.
* U.S. Department of Energy. Form OE-417, Electric Emergency Incident and Disturbance Report (annual summaries).
