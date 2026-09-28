"""Export D02 descriptive figures and compact checks from public summaries only."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/v1'
COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']
ANCHORS = ['gust', 'precip', 'cold', 'snowfall', 'soil', 'cape']
LABELS = ['Gust', 'Precipitation', 'Cold (-temperature)', 'Snowfall', 'Soil moisture', 'CAPE']
REGIMES = ['tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain']


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def load(name):
    return json.loads((OUT / name).read_text())


def check_and_extract(dynamics):
    assert set(dynamics['weather_anchors']) == set(ANCHORS)
    extracted = {}
    checked = 0
    for anchor, a in dynamics['weather_anchors'].items():
        assert sum(p['support']['county_events'] for p in a['by_type'].values()) == 8457
        assert sum(p['support']['county_events'] for p in a['by_regime'].values()) == 8457
        assert len(a['by_type_regime']) == 30 and len(a['by_type_intensity']) == 18
        one = {}
        for name, p in [('overall', a['overall']), *a['by_type'].items(), *a['by_regime'].items(),
                        *a['by_type_regime'].items(), *a['by_type_intensity'].items()]:
            offsets = arr(p['offsets_hours'])
            for metric, c in p['curves'].items():
                mean, lo, hi = [arr(c[k]) for k in ['mean', 'ci95_lower', 'ci95_upper']]
                assert len(mean) == len(lo) == len(hi) == len(offsets)
                paired = np.isfinite(lo) & np.isfinite(hi)
                assert np.all(lo[paired] <= hi[paired])
                assert np.all(arr(c['observed_county_events']) <= p['aligned_support']['county_events'])
                assert np.all(arr(c['observed_event_groups']) <= p['aligned_support']['merged_event_groups'])
                assert np.all((arr(c['observed_weight_fraction']) >= 0) &
                              (arr(c['observed_weight_fraction']) <= 1.000001))
                if metric == 'outage_stock':
                    assert np.all((mean[np.isfinite(mean)] >= 0) & (mean[np.isfinite(mean)] <= 1))
                checked += 1
            if name in a['by_type']:
                y = arr(p['curves']['outage_stock']['mean'])
                dy = arr(p['curves']['net_stock_change_per_hour']['mean'])
                j = np.nanargmax(y) if np.isfinite(y).any() else None
                one[name] = dict(support=p['aligned_support'], statuses=p['statuses'],
                    stock_curve_peak_offset_hours=None if j is None else int(offsets[j]),
                    stock_curve_peak_percent=None if j is None else float(100*y[j]),
                    net_change_at_weather_peak_percentage_points=float(100*dy[offsets == 0][0]),
                    complete_interior_peak_lag_quantiles_hours=p['weather_peak_lag']['complete_forecast_interior_peaks']['midpoint_hours']['quantiles_10_25_50_75_90'],
                    complete_interior_peak_lag_support=p['weather_peak_lag']['complete_forecast_interior_peaks']['support'])
        extracted[anchor] = one
    return {'curve_fields_checked': checked, 'partitions_and_support_checks_passed': True,
            'interpretation': 'Peak of cohort mean is distinct from distribution of individual peak lags. '
            'Type comparisons do not control storm composition or exposure.',
            'descriptive_type_patterns': extracted}


def dynamic_figure(d, regime=None):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.6), sharex=True, sharey='row')
    for j, anchor in enumerate(ANCHORS):
        a = d['weather_anchors'][anchor]
        for k in range(6):
            p = a['by_type'][f'T{k}'] if regime is None else a['by_type_regime'][f'T{k}/{regime}']
            x = arr(p['offsets_hours'])
            for row, metric in enumerate(['outage_stock', 'net_stock_change_per_hour']):
                ax = axes[row, j]
                c = p['curves'][metric]
                y, lo, hi = [100*arr(c[key]) for key in ['mean', 'ci95_lower', 'ci95_upper']]
                ax.plot(x, y, lw=1.2, color=COLORS[k], label=f'T{k}')
                ax.fill_between(x, lo, hi, color=COLORS[k], alpha=.065, linewidth=0)
        for row in range(2):
            ax = axes[row, j]
            ax.axvline(0, color='.55', lw=.7)
            if row == 1:
                ax.axhline(0, color='.55', lw=.7)
                ax.set_xlabel('Hours from weather peak')
            ax.set_xticks([-24, 0, 24])
            ax.spines[['top', 'right']].set_visible(False)
        axes[0, j].set_title(LABELS[j], fontsize=11)
    axes[0, 0].set_ylabel('Observed outage stock (%)')
    axes[1, 0].set_ylabel('Net stock change (pp/hour)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc='upper center', bbox_to_anchor=(.5, .98), frameon=False)
    label = 'All regimes (composition unadjusted)' if regime is None else regime.replace('_', ' ').title()
    fig.suptitle(f'D02 county structure: weather-aligned responses — {label}', y=1.02, fontsize=13)
    fig.text(.02, .01, 'Weather-only anchors; missing cells excluded with changing denominators. '
             'Bands: descriptive event-group bootstrap 95% intervals. Types are coarse continuous-structure partitions.', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .93])
    suffix = 'all' if regime is None else regime
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'fig_county_d02_dynamics_{suffix}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def association_figures(d):
    """Full driver space, with no significance filter or coefficient clipping."""
    a = np.asarray(d['estimates'], dtype=float)
    assert a.shape == (5, 6, 56, 4, 2, 3, 3)
    for response, name in enumerate(d['meta']['responses']):
        panels = [a[:, :, :, 0, 0, response, 0].transpose(2, 0, 1).reshape(56, 30),
                  a[:, :, :, 3, 1, response, 0].transpose(2, 0, 1).reshape(56, 30)]
        limit = float(np.nanmax(np.abs(panels)))
        norm = SymLogNorm(linthresh=.02, vmin=-limit, vmax=limit, base=10)
        fig, axes = plt.subplots(1, 2, figsize=(18, 18), sharey=True)
        cmap = plt.colormaps['RdBu_r'].copy()
        cmap.set_bad('.82')
        for ax, values, title in zip(axes, panels, ['Unadjusted association', 'County + system/phase + starting stock/history']):
            im = ax.imshow(values, aspect='auto', cmap=cmap, norm=norm, interpolation='none')
            ax.set_xticks(np.arange(30), [f'T{k}' for _ in range(5) for k in range(6)], fontsize=7)
            ax.xaxis.tick_top()
            ax.tick_params(axis='x', length=0, pad=2)
            for k, regime in enumerate(REGIMES):
                ax.text(k*6+2.5, -2.3, regime.replace('_', '\n'), ha='center', va='bottom', fontsize=8)
                if k:
                    ax.axvline(k*6-.5, color='.5', lw=.6)
            for row in [11.5, 23.5, 38.5, 53.5]:
                ax.axhline(row, color='.5', lw=.6)
        axes[0].set_yticks(np.arange(56), d['meta']['drivers'], fontsize=8)
        fig.subplots_adjust(left=.19, right=.985, top=.89, bottom=.12, wspace=.065)
        for ax, title in zip(axes, ['Unadjusted association', 'County + system/phase + starting stock/history']):
            box = ax.get_position()
            fig.text((box.x0+box.x1)/2, .958, title, ha='center', fontsize=11)
        bar = fig.colorbar(im, cax=fig.add_axes([.25, .066, .67, .012]), orientation='horizontal')
        bar.set_label('Response percentage points per regime-wide driver SD; symmetric-log color scale, linear within ±0.02', fontsize=9)
        fig.suptitle(f'D02 complete association atlas — {name.replace("_", " ")}', y=.992, fontsize=14)
        fig.text(.025, .022, 'Each cell is a separate descriptive association. Grey = undefined. No multiple-comparison adjustment. '
                 'Intervals, weather support, event leverage and all other specifications are retained in the result JSON.', fontsize=9)
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'fig_county_d02_associations_{name}.{ext}', dpi=150, bbox_inches='tight')
        plt.close(fig)


def review_associations(d):
    root = HERE.parents[1]
    hashes = {}
    for relative, expected in d['meta']['frozen_sha256'].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
        hashes[relative] = actual
    a = np.asarray(d['estimates'], float)
    assert a.shape == (5, 6, 56, 4, 2, 3, 3)
    assert d['projection_diagnostics']['failed'] == 0
    q = np.asarray(d['type_q'])
    assert q.shape == (5, 6) and np.allclose(q.sum(1), 1) and (q > 0).all()
    checked = 0
    for field in ['estimates', 'common_support_estimates', 'county_event_aggregate_estimates', 'wind_speed_adjusted_estimates']:
        value = np.asarray(d[field], float)
        beta, lo, hi = np.moveaxis(value, -1, 0)
        good = np.isfinite(lo) & np.isfinite(hi)
        assert np.all(lo[good] <= beta[good]) and np.all(beta[good] <= hi[good]), field
        checked += int(good.sum())
    event = np.asarray(d['county_event_aggregate_estimates'], float)
    assert np.isnan(event[:, :, -2:]).all(), 'Unmatched terrain window support must not be silently averaged'
    slopes = a[..., 0]
    active = np.isfinite(slopes).all(axis=1)
    weighted = q[:, :, None, None, None, None]*slopes
    signed = weighted.sum(1)
    absolute = np.abs(weighted).sum(1)
    computed = 1-np.divide(np.abs(signed), absolute, out=np.full_like(signed, np.nan), where=absolute > 1e-10)
    saved = np.asarray(d['full_domain_cancellation'], float)
    # JSON is deliberately rounded to seven significant digits.
    assert np.allclose(saved[..., 0][active], signed[active], atol=2e-6, rtol=2e-5, equal_nan=True)
    assert np.allclose(saved[..., 2][active], computed[active], atol=2e-6, rtol=2e-5, equal_nan=True)
    examples = {}
    for regime, driver in [('tropical','mean:gust'), ('heavy_rain','mean:precip'), ('tropical','order:gust->precip')]:
        ri = d['meta']['regimes'].index(regime)
        j = d['meta']['drivers'].index(driver)
        examples[regime+'/'+driver] = {
            'illustrative_not_selected_for_confirmation': True,
            'response': 'mean_stock_change',
            'units': d['meta']['response_units'],
            'unadjusted_by_type': a[ri, :, j, 0, 0, 0].tolist(),
            'two_way_stock_history_by_type': a[ri, :, j, 3, 1, 0].tolist(),
            'common_interval_two_way_stock_history_by_type': np.asarray(d['common_support_estimates'])[ri, :, j, 1, 0].tolist(),
            'residual_support_by_type': np.asarray(d['residual_driver_diagnostics'])[ri, :, j, 3, 1].tolist(),
            'common_interval_residual_support_by_type': np.asarray(d['common_support_residual_driver_diagnostics'])[ri, :, j, 1].tolist(),
        }
    out = {'source_hashes_match': hashes, 'estimate_shape': list(a.shape), 'ci_entries_checked': checked,
           'undefined_slopes_retained': int((~np.isfinite(a[..., 0])).sum()),
           'projection_diagnostics': d['projection_diagnostics'],
           'cancellation_recomputation_passed': True, 'terrain_event_support_guard_passed': True,
           'examples': examples}
    (OUT / 'county_d02_association_review.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')


def main():
    d = load('county_dynamics_d02.json')
    audit = check_and_extract(d)
    (OUT / 'county_d02_dynamics_review.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    for regime in [None, *REGIMES]:
        dynamic_figure(d, regime)
    path = OUT / 'county_heterogeneity_d02.json'
    if path.exists():
        h = load(path.name)
        review_associations(h)
        association_figures(h)
    print(json.dumps({'curves_checked': audit['curve_fields_checked'], 'dynamics_figures': 6,
                      'association_figures': 3 if path.exists() else 0}))


if __name__ == '__main__':
    main()
