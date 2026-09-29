"""Render the completed D08 fixed-panel audit; no new fitting or scoring."""
from pathlib import Path
import json
import gzip
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results/v1'
RUNTIME = ROOT / 'runs/geo_weather_20260924/d08_panel_20260929_iofix1'


def main():
    out = RESULTS / 'fig_d08_fixed_panel'
    if out.with_suffix('.png').exists() or out.with_suffix('.pdf').exists():
        raise FileExistsError('Preserve earlier figures')
    inputs = json.loads((RESULTS / 'd08_panel_input.json').read_text())
    packaged = RESULTS / 'd08_panel_symptoms.json.gz'
    report = json.loads(gzip.decompress(packaged.read_bytes()) if packaged.exists()
                        else (RESULTS / 'd08_panel_symptoms.json').read_bytes())
    marker = json.loads((RUNTIME / 'FROZEN.json').read_text())
    assert marker['status'] == 'input_roster_frozen'
    assert report['provenance']['roster_sha256'] == marker['roster_sha256']
    arrays = RUNTIME / 'symptoms_arrays.npz'
    assert hashlib.sha256(arrays.read_bytes()).hexdigest() == report['provenance']['local_arrays_sha256']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7), constrained_layout=True)
    scopes = ['full_original_fold1_FIT', 'frozen_input_panel']
    colors = ['#687991', '#176b8c']
    ax = axes[0, 0]
    labels = ['Weather path', 'Hourly change', 'Joint geo40', 'Joint input']
    keys = ['weather_full216_level_and_change', 'hourly_change_full215', 'geography_all40', 'joint']
    q = np.array([inputs['coverage'][k]['equal_group_design']['median_q90_max'] for k in keys])
    ax.barh(np.arange(4), q[:, 0], color=colors[1], alpha=.7, label='Median')
    ax.scatter(q[:, 1], np.arange(4), c='#cc7b2e', marker='|', s=150, label='90th percentile')
    ax.set_yticks(np.arange(4), labels)
    ax.set_xlabel('Nearest selected point distance, within group')
    ax.set_title('A. Input coverage, all 53 groups')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for j, scope in enumerate(scopes):
        rows = report['scores'][scope]['common_observed']
        changes = [(rows[c]['models']['crk']['local_design_RMSE'] /
                    rows[c]['models']['host']['local_design_RMSE'] - 1) * 100
                   for c in ['S', 'nonS', 'all']]
        ax.bar(np.arange(3) + (j-.5)*.34, changes, .34, color=colors[j],
               label=['Full FIT', 'Fixed panel'][j])
    ax.axhline(0, color='black', lw=.7)
    ax.set_xticks(np.arange(3), ['S', 'non-S', 'All'])
    ax.set_ylabel('CRK vs W RMSE change (%)')
    ax.set_title('B. Frozen FIT response allocation')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    for j, scope in enumerate(scopes):
        rows = report['scores'][scope]['common_observed']['S']['models']
        vals = [rows[name]['local_peak_ratio']['q10_median_q90'] for name in ['host', 'crk']]
        x = np.arange(2) + (j-.5)*.18
        ax.scatter(x, [v[1]*100 for v in vals], color=colors[j], label=['Full FIT median', 'Panel median'][j])
        for xx, v in zip(x, vals):
            ax.plot([xx, xx], [max(v[0]*100, .001), v[2]*100], color=colors[j], alpha=.55)
    ax.set_yscale('log')
    ax.set_xticks([0, 1], ['W', 'CRK'])
    ax.set_ylabel('Predicted / observed peak (%)')
    ax.set_title('C. S peak ratio; line = 10th to 90th')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    table = report['severe_count_vs_contiguous_run']['frozen_input_panel']
    matrix = np.array([r['units'] for r in table]).reshape(3, 3)
    ax.imshow(matrix, cmap='Blues', aspect='auto')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
    labels = ['1', '2–6', '≥7']
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel('Longest consecutive severe run (hours)')
    ax.set_ylabel('Total severe hours')
    ax.set_title('D. Selected S: count differs from continuity')

    ax = axes[1, 1]
    paired = report['frozen_input_pair_outcome_descriptives']['rows']
    for j, key in enumerate(['near_weather_far_geo', 'near_geo_far_weather']):
        dist = paired[key]['unweighted_descriptive_distributions']
        names = ['true_trajectory_RMS_difference', 'host_trajectory_RMS_difference',
                 'crk_trajectory_RMS_difference']
        values = [dist[k]['mean']*100 for k in names]
        ax.bar(np.arange(3) + (j-.5)*.34, values, .34, color=colors[j],
               label=['Weather near / geo far', 'Geo near / weather far'][j])
    ax.set_xticks(np.arange(3), ['Observed', 'W', 'CRK'])
    ax.set_ylabel('Mean endpoint trajectory difference (pp)')
    ax.set_title('E. Frozen input contrasts; dependent pairs')
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    pair = next(p for p in inputs['pairs'] if p['status'] == 'selected' and p['kind'] == 'near_weather_far_geo')
    with np.load(arrays, allow_pickle=False) as z:
        unit = z['unit']
        ids = [int(np.flatnonzero(unit == pair[k])[0]) for k in ['anchor_unit', 'partner_unit']]
        for j, row in enumerate(ids):
            observed = np.where(z['m'][row], z['y'][row]*100, np.nan)
            ax.plot(np.arange(1, 145), observed, color=colors[j], lw=1.6,
                    label=f'Observed unit {unit[row]}')
            ax.plot(np.arange(1, 145), z['P_crk'][row]*100, color=colors[j], lw=.9, ls='--',
                    label=f'CRK unit {unit[row]}')
    ax.set_xlabel('Hours from forecast origin')
    ax.set_ylabel('Outage fraction (%)')
    ax.set_title('F. First input-selected pair\nNo outcome selection')
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle('D08: fixed mechanism panel, existing FIT-trained models\n'
                 '1,219 county-events / 53 groups; descriptive replay, no new event generalization', fontsize=12)
    fig.savefig(out.with_suffix('.png'), dpi=180)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
