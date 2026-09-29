"""Static D05 figures from the five compact result JSONs only.

No panel, prediction array, model, bootstrap or analysis module is loaded. The
within-county PDF includes the pooled page and all five regime pages; its PNG
shows the pooled page. Missing support is never plotted as zero. Run --execute
after the owning D05 analysis has finished, or --self-test for synthetic inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results/v1'
INPUTS = ('phenotypes', 'frozen_models', 'aligned_curves', 'within_county', 'geography_matches')
REGIMES = ('tropical', 'winter', 'synoptic_wind', 'convective', 'heavy_rain')
REGIME_LABELS = ('Tropical', 'Winter', 'Synoptic wind', 'Convective', 'Heavy rain')
WEATHER = ('cape', 'cloud', 'gust', 'precip', 'pressure', 'rh', 'snowfall',
           'soil_moisture', 't2m_c', 'u10', 'v10', 'wind_speed')
WX_LABELS = ('CAPE [log]', 'Cloud cover', 'Hourly maximum gust', 'Precipitation [log]',
             'Surface pressure', 'Relative humidity', 'Snowfall [log]', 'Soil moisture',
             'Temperature', 'Eastward wind', 'Northward wind', 'Wind speed')
MODELS = ('v1_host_s0', 'v1_gcrk_s0', 'v1_gcrk_georms_s0')
MODEL_LABELS = ('Host', 'GCRK', 'Geo-RMS')
MODEL_COLORS = ('#34495e', '#0072b2', '#d55e00')
REGIME_COLORS = ('#0072b2', '#7b4ab5', '#009e73', '#d55e00', '#5c677d')
WINDOWS = ('offset_-48_-25', 'offset_-24_-7', 'offset_-6_-1')
WINDOW_LABELS = ('Mean: -48 to -25 h', 'Mean: -24 to -7 h', 'Mean: -6 to -1 h',
                 'Near: -1 h', 'Near: 0 h', 'Near: +1 h')
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
         'axes.spines.top': False, 'axes.spines.right': False,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    return np.asarray(value, dtype=float)  # explicit JSON null becomes NaN


def _summary(value, width):
    mean = _array(value['mean'])
    _require(mean.shape == (width,), f'Expected a flat mean vector of length {width}')
    ci = _array(value['ci95']) if 'ci95' in value else np.full((2, width), np.nan)
    _require(ci.shape == (2, width), 'Incorrect confidence-interval shape')
    valid_ci = np.isfinite(ci).all(0)
    _require(np.all(ci[0, valid_ci] <= ci[1, valid_ci]), 'Reversed confidence bounds')
    n = _array(value.get('n', np.full(width, np.nan)))
    _require(n.shape == (width,), 'Incorrect support-count shape')
    _require(not np.isinf(mean).any() and not np.isinf(ci).any(), 'Infinite summary statistic')
    return mean, ci, n


def _read_reports(directory):
    reports, digests = {}, {}
    for name in INPUTS:
        path = directory/f'd05_{name}.json'
        raw = path.read_bytes()
        reports[name] = json.loads(raw)
        _require(isinstance(reports[name], dict), f'{path.name} is not an object')
        digests[path.name] = hashlib.sha256(raw).hexdigest()
    return reports, digests


def _title(title, synthetic):
    return ('SYNTHETIC TEST ONLY | ' if synthetic else '') + title


def _save_pair(fig, stem, title):
    metadata = {'Creator': 'report_d05_forensics.py', 'Title': title,
                'Subject': 'Derived only from compact D05 summaries; no new fitting or resampling.'}
    fig.savefig(stem.with_suffix('.png'), dpi=170, metadata=metadata)
    fig.savefig(stem.with_suffix('.pdf'), metadata=metadata)


def _phenotype_and_models(phenotype, frozen, synthetic=False):
    counts = phenotype['counts']
    keys = ('J_only', 'J_and_S', 'S_only', 'neither')
    values = np.array([counts[k] for k in keys], dtype=int)
    _require(np.all(values >= 0) and values.sum() == counts['all'], 'Phenotype cells do not partition all units')
    _require(counts['J'] == counts['J_only']+counts['J_and_S']
             and counts['S'] == counts['S_only']+counts['J_and_S'], 'Phenotype overlap counts disagree')
    fields = ('at_true_anchor_ratio', 'max_on_observed_support_ratio', 'max_within_6h_ratio')
    ratios = np.empty((3, 3))
    for i, model in enumerate(MODELS):
        q = frozen['models'][model]['stock_max']['weighted_q10_50_90']
        for j, name in enumerate(fields):
            vals = _array(q[name])
            _require(vals.shape == (3,), f'Expected q10/q50/q90 for {model}/{name}')
            ratios[i, j] = vals[1]
    _require(not np.isinf(ratios).any(), 'Infinite peak ratio')

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), gridspec_kw={'width_ratios': [1, 1.5]})
    ax = axes[0]
    bars = ax.bar(np.arange(4), values, color=['#56b4e9', '#0072b2', '#e69f00', '#aeb7c0'], width=.7)
    for bar, value in zip(bars, values):
        ax.annotate(f'{value:,}', (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(np.arange(4), ['J only', 'J and S', 'S only', 'Neither'])
    ax.set_ylim(0, max(1, float(values.max()))*1.2)
    ax.set_ylabel('County-events (unweighted count)')
    ax.set_title(f'A  Distinct observed phenotypes | n = {counts["all"]:,}', loc='left', fontweight='bold')
    ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
    ax.text(0, -.20, 'J: largest one-hour net increase >= 1 pp\nS: largest observed outage stock >= 10%',
            transform=ax.transAxes, va='top', fontsize=9)

    ax = axes[1]
    x = np.arange(3); width = .23
    markers = ('o', 's', '^')
    finite = ratios[np.isfinite(ratios)]
    use_log = len(finite) > 0 and bool(np.all(finite > 0))
    for i, (label, color) in enumerate(zip(MODEL_LABELS, MODEL_COLORS)):
        xs = x + (i-1)*width
        ax.scatter(xs, ratios[i], color=color, label=label, marker=markers[i], s=40, zorder=3)
        for xpos, value in zip(xs, ratios[i]):
            if np.isfinite(value):
                ax.annotate(f'{value:.4f}x', (xpos, value), xytext=(0, 8),
                            textcoords='offset points', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(xpos, .02, 'N/A', ha='center', fontsize=8)
    if use_log:
        ax.set_yscale('log')
        ax.set_ylim(min(float(finite.min())*.48, .003), max(1.4, float(finite.max())*1.5))
        ticks = np.asarray([.001, .003, .01, .03, .1, .3, 1, 3, 10])
        ticks = ticks[(ticks >= ax.get_ylim()[0]) & (ticks <= ax.get_ylim()[1])]
        ax.set_yticks(ticks, [f'{v:g}x' for v in ticks])
        ax.minorticks_off()
    else:
        ax.set_ylim(0, max(1.12, float(finite.max())*1.25 if len(finite) else 1.12))
    ax.axhline(1, color='#777777', ls='--', lw=1)
    ax.set_xticks(x, ['At observed stock peak', 'Full forecast-window max\n(observed support)', 'Maximum within +/-6 h'])
    ax.set_ylabel('Predicted / observed peak stock (ratio'+('; log scale)' if use_log else ')'))
    ax.set_title(f'B  Frozen forecasts on S cases | n = {counts["S"]:,}', loc='left', fontweight='bold')
    ax.legend(frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, .89), fontsize=9)
    ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
    ax.text(0, -.20, 'Points: original-design-weight median of per-case ratios. 1x = the observed peak amplitude.\n'
            'Maxima use the common observed support. A ratio is not a percent error change; medians have no CI here.',
            transform=ax.transAxes, va='top', fontsize=8.5)
    fig.suptitle(_title('D05 | Large stock and large increases are different case sets', synthetic),
                 x=.06, ha='left', fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=.065, right=.985, top=.83, bottom=.25, wspace=.32)
    return fig


def _curve_data(curves, regime):
    value = curves['cohorts']['jump_max'].get(regime)
    if value is None:
        return None
    offsets = _array(curves['offsets'])
    columns = list(value['columns'])
    _require(value['shape'] == [len(offsets), len(columns)], 'Curve shape metadata disagrees')
    _require(columns == [*WEATHER, 'stock', 'net_change'], 'Unexpected weather/outage curve order')
    mean, ci, n = _summary(value['summary'], len(offsets)*len(columns))
    return (mean.reshape(len(offsets), len(columns)),
            ci.reshape(2, len(offsets), len(columns)), n.reshape(len(offsets), len(columns)), value['n_units'])


def _aligned_atlas(curves, synthetic=False):
    offsets = _array(curves['offsets'])
    _require(np.array_equal(offsets, np.arange(-48, 25)), 'Expected registered offsets -48..24')
    _require(tuple(curves['weather_names']) == WEATHER, 'Expected all twelve registered weather channels')
    data = [_curve_data(curves, r) for r in REGIMES]
    fig, axes = plt.subplots(14, 5, figsize=(15.4, 27.2), sharex=True, sharey='row')
    labels = [*WX_LABELS, 'Observed stock', 'One-hour net change']
    for j, (regime, label, color, value) in enumerate(zip(REGIMES, REGIME_LABELS, REGIME_COLORS, data)):
        for i in range(14):
            ax = axes[i, j]
            ax.axvspan(0, 24, color='#f4f1e7', zorder=0)
            ax.axhline(0, color='#d1d5da', lw=.7, zorder=0)
            ax.axvline(0, color='#777777', ls=':', lw=.8)
            if value is None:
                ax.text(.5, .5, 'No eligible cases', transform=ax.transAxes, ha='center', color='#777777', fontsize=8)
            else:
                mean, ci, n, count = value
                scale = 100 if i >= 12 else 1
                ax.fill_between(offsets, ci[0, :, i]*scale, ci[1, :, i]*scale,
                                color=color, alpha=.16, linewidth=0)
                ax.plot(offsets, mean[:, i]*scale, color=color, lw=1.2)
                finite_n = n[:, i][np.isfinite(n[:, i])]
                support = (f'n {int(finite_n.min())}-{int(finite_n.max())}' if len(finite_n) else 'n unknown')
                if not np.isfinite(ci[:, :, i]).any(): support += '; CI unavailable'
                ax.text(.03, .96, support, transform=ax.transAxes, va='top', fontsize=6.5, color='#555555')
            if i == 0:
                nlabel = f' | {value[3]:,} cases' if value is not None else ''
                ax.set_title(label+nlabel, fontsize=9, fontweight='bold', pad=9)
            if j == 0:
                ax.set_ylabel(labels[i]+'\n'+('percentage points' if i >= 12 else 'weather z'), fontsize=8.2)
            if i == 13: ax.set_xlabel('Hours from largest net increase', fontsize=8)
            ax.set_xticks([-48, -24, 0, 24]); ax.tick_params(labelsize=7, length=3)
    fig.suptitle(_title('D05 | Weather and outage trajectories around the largest >=1 pp increase', synthetic),
                 x=.08, ha='left', fontsize=14, fontweight='bold', y=.992)
    fig.text(.08, .974, 'Every registered weather channel and all five regimes; columns share a scale within each row.', fontsize=9)
    fig.text(.08, .025,
             'Lines: original-design-weight means. Bands: pointwise 95% merged-event bootstrap conditional on observed anchors and reference.\n'
             'CAPE, precipitation and snowfall are log1p transformed before standardization; n shows contributing cases across offsets.\n'
             'Stock and net change are shown in percentage points. The shaded region is contemporaneous/retrospective, not forecast information.\n'
             'Precipitation, snowfall and gust at T cover (T-1,T]; other weather refers to T. Outage p[T] averages collection starts T through T+45 min.\n'
             'These outcome-aligned descriptions neither establish causality nor adjust for multiple comparisons.', fontsize=8, va='bottom')
    fig.subplots_adjust(left=.105, right=.985, top=.955, bottom=.071, hspace=.21, wspace=.16)
    return fig


def _within_values(within, cohort, group):
    value = within['cohorts'][cohort]
    names = list(value['feature_names'])
    _require(len(set(names)) == len(names), 'Duplicate within-county feature names')
    wanted = [[f'standardized:mean:{wx}:{window}' for window in WINDOWS]
              + [f'near_z:{offset}:{wx}' for offset in (-1, 0, 1)] for wx in WEATHER]
    pos = {name: i for i, name in enumerate(names)}
    _require(all(name in pos for row in wanted for name in row), 'Missing registered weather/time feature')
    if group not in value['groups']:
        return None
    mean, ci, n = _summary(value['groups'][group], len(names))
    idx = np.asarray([[pos[name] for name in row] for row in wanted])
    return mean[idx], ci[:, idx], n[idx]


def _within_page(within, group, synthetic=False):
    js = [_within_values(within, cohort, group) for cohort in ('jump_max', 'stock_max')]
    group_name = 'All regimes pooled' if group == 'all' else REGIME_LABELS[REGIMES.index(group)]
    fig, axes = plt.subplots(4, 3, figsize=(15.7, 15.5))
    for i, (ax, label) in enumerate(zip(axes.flat, WX_LABELS)):
        ax.axhspan(3.5, 5.5, color='#f4f1e7', zorder=0)
        ax.axvline(0, color='#8a8a8a', ls=':', lw=1)
        for k, (data, color, marker) in enumerate(zip(js, ('#0072b2', '#d55e00'), ('o', 's'))):
            if data is None: continue
            mean, ci, n = data
            yy = np.arange(6)+(k-.5)*.20
            finite = np.isfinite(mean[i])
            interval = finite & np.isfinite(ci[:, i]).all(0)
            # Draw the supplied interval endpoints directly. Percentile intervals
            # need not contain the point estimate, so symmetric xerr is unsuitable.
            ax.hlines(yy[interval], ci[0, i, interval], ci[1, i, interval], color=color, lw=1.4, alpha=.8)
            ax.plot(mean[i, finite], yy[finite], marker, color=color, markersize=4,
                    label=('J: largest increase' if k == 0 else 'S: largest stock'))
            for row in np.flatnonzero(~finite):
                ax.text(.98, (5.7-yy[row])/6.4, ('J' if k == 0 else 'S')+': N/A',
                        transform=ax.transAxes, ha='right', color=color, fontsize=6.5)
        ax.set_title(label, loc='left', fontweight='bold', fontsize=10)
        ax.set_ylim(5.7, -.7)
        ax.set_yticks(np.arange(6), WINDOW_LABELS, fontsize=7.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='x', alpha=.15)
        if i >= 9: ax.set_xlabel('Case - same-event control mean (weather z)', fontsize=8)
        if all(value is None for value in js):
            ax.text(.5, .5, 'No eligible cases', transform=ax.transAxes, ha='center', color='#777777')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, .985), frameon=False, ncol=2)
    fig.suptitle(_title('D05 | Within-county/event weather contrasts | '+group_name, synthetic),
                 x=.065, ha='left', fontsize=14, fontweight='bold', y=.995)
    fig.text(.065, .966, 'All 12 channels x three strictly past bands and -1/0/+1 h; points are design-weighted paired means, lines are 95% CI.', fontsize=9)
    fig.text(.065, .024,
             'Controls: mean of eligible fixed-clock anchors in the same county-event, more than 12 h from the observed case anchor; no quiet-outcome filter.\n'
             'Intervals: pointwise merged-event bootstrap conditional on anchors/reference; neither fitting uncertainty nor simultaneous confidence bands.\n'
             'Shaded near-0/+1 rows are not strictly-past forecasting information. +1 h accumulated weather/gust retrospectively covers the observation hour.\n'
             'Support varies by feature; N/A is not zero. Full coordinate-level support and all other predeclared features remain in the JSON.\n'
             'The PNG shows the pooled page. This PDF also contains every regime page; no channel or interval was selected by statistical significance.', fontsize=8.2)
    fig.subplots_adjust(left=.135, right=.975, top=.934, bottom=.095, hspace=.33, wspace=.58)
    return fig


def render_reports(input_dir=RESULTS, output_dir=RESULTS, *, overwrite=False, synthetic=False):
    """Read the five compact JSONs and write three PNG/PDF figure pairs."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    reports, hashes = _read_reports(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'fig_d05_synthetic_' if synthetic else 'fig_d05_'
    stems = [output_dir/(prefix+name) for name in
             ('phenotypes_frozen_peaks', 'jump_aligned_weather', 'within_county_weather')]
    paths = [stem.with_suffix(ext) for stem in stems for ext in ('.png', '.pdf')]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Preserve existing figures; use --overwrite explicitly after review')
    with plt.rc_context(STYLE):
        fig = _phenotype_and_models(reports['phenotypes'], reports['frozen_models'], synthetic)
        _save_pair(fig, stems[0], 'D05 phenotype counts and frozen stock-peak ratios'); plt.close(fig)
        fig = _aligned_atlas(reports['aligned_curves'], synthetic)
        _save_pair(fig, stems[1], 'D05 all-channel event-aligned weather/outage atlas'); plt.close(fig)
        with PdfPages(stems[2].with_suffix('.pdf'), metadata={
                'Title': 'D05 within-county weather contrasts: pooled and every regime',
                'Creator': 'report_d05_forensics.py',
                'Subject': 'Pointwise intervals from existing JSON only; no new statistics.'}) as pdf:
            for group in ('all', *REGIMES):
                fig = _within_page(reports['within_county'], group, synthetic)
                if group == 'all':
                    fig.savefig(stems[2].with_suffix('.png'), dpi=170,
                                metadata={'Title': 'D05 pooled within-county weather contrasts'})
                pdf.savefig(fig); plt.close(fig)
    return {'input_json_sha256': hashes, 'files': [str(path) for path in paths],
            'within_county_pdf_pages': ['all', *REGIMES], 'synthetic_only': synthetic,
            'ratio_unit': 'Dimensionless predicted/observed ratio, not relative percent change',
            'geographic_matching': 'Read for input provenance; no new geographic-effect claim is plotted'}


def _json_clean(value):
    if isinstance(value, dict): return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [_json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def _synthetic_reports():
    counts = {'all': 120, 'J': 50, 'S': 40, 'J_and_S': 25, 'J_only': 25, 'S_only': 15, 'neither': 55}
    models = {}
    for i, model in enumerate(MODELS):
        q = {key: [v*.6, v, v*1.5] for key, v in zip(
            ('at_true_anchor_ratio', 'max_on_observed_support_ratio', 'max_within_6h_ratio'),
            np.array([.18, .46, .34])+.04*i)}
        models[model] = {'stock_max': {'weighted_q10_50_90': q}}
    curves = {'offsets': np.arange(-48, 25), 'weather_names': WEATHER, 'cohorts': {'jump_max': {}}}
    for j, regime in enumerate(REGIMES):
        t = np.arange(-48, 25)
        x = np.stack([.12*j+np.sin(t/(9+i)+i)*(.15+.03*i) for i in range(12)], axis=1)
        x = np.column_stack([x, .08+.12/(1+np.exp(-t/3)), .025*np.exp(-(t/2)**2)])
        half = np.r_[np.full(12, .08), .012, .004]
        ci = np.stack([x-half, x+half])
        n = np.full_like(x, 50+j)
        if j == 4: x[-2:] = np.nan; ci[:, -2:] = np.nan; n[-2:] = 0
        summary = {'mean': x.ravel(), 'ci95': ci.reshape(2, -1), 'n': n.ravel()}
        curves['cohorts']['jump_max'][regime] = {'n_units': 50+j, 'shape': [73, 14],
                'columns': [*WEATHER, 'stock', 'net_change'], 'summary': summary}
    names = [f'standardized:mean:{wx}:{band}' for band in WINDOWS for wx in WEATHER]
    names += [f'near_z:{off}:{wx}' for off in (-1, 0, 1) for wx in WEATHER]
    names = list(np.random.default_rng(17).permutation(names))  # test label lookup rather than column offsets
    within = {'cohorts': {}}
    for k, cohort in enumerate(('jump_max', 'stock_max')):
        groups = {}
        for j, group in enumerate(('all', *REGIMES)):
            mean = .3*np.sin(np.arange(len(names))*.25+j)+k*.1
            ci = np.stack([mean-.12, mean+.12]); n = np.full(len(names), 45, dtype=float)
            if k == 1: mean[0] = np.nan; ci[:, 0] = np.nan; n[0] = 0
            groups[group] = {'mean': mean, 'ci95': ci, 'n': n}
        within['cohorts'][cohort] = {'feature_names': names, 'groups': groups}
    return {'phenotypes': {'counts': counts}, 'frozen_models': {'models': models},
            'aligned_curves': curves, 'within_county': within,
            'geography_matches': {'synthetic_only': True}}


def self_test(output_dir):
    output_dir = Path(output_dir)
    _require(output_dir.resolve() != RESULTS.resolve(), 'Synthetic plots cannot go in the real result directory')
    _require(not output_dir.exists(), 'Use a new directory for the synthetic render test')
    input_dir = output_dir/'synthetic_json'; input_dir.mkdir(parents=True)
    reports = _synthetic_reports()
    for name, value in reports.items():
        (input_dir/f'd05_{name}.json').write_text(json.dumps(_json_clean(value), allow_nan=False))
    result = render_reports(input_dir, output_dir, synthetic=True)
    for file in result['files']:
        path = Path(file)
        _require(path.stat().st_size > 1000, 'Empty synthetic figure')
        if path.suffix == '.pdf': _require(path.read_bytes().startswith(b'%PDF'), 'Invalid PDF header')
        else:
            image = plt.imread(path)
            _require(image.shape[0] > 500 and image.shape[1] > 500 and image.std() > .03, 'Blank raster output')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--input-dir', type=Path, default=RESULTS)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        if args.output_dir is None: parser.error('--self-test requires a new --output-dir outside real results')
        result = self_test(args.output_dir)
    else:
        result = render_reports(args.input_dir, args.output_dir or RESULTS, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
