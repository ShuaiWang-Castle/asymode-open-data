"""US data-contract gates on the real frozen corpus (loads it once, ~40 s)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch

W = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def ctx():
    import us_train as UT
    return UT, UT.USData(ROOT, W / 'locks')


def test_splits_lock_events_and_no_leakage():
    s = json.loads((W / 'locks/SPLITS.json').read_text())
    assert s['events_match_expected'] and s['cross_split_leakage'] == 'NONE'
    assert [s['splits'][k]['n_events'] for k in ('train', 'validation', 'evaluation')] == [14, 6, 6]


def test_states_are_exact_hourly_snapshots_not_means(ctx):
    UT, d = ctx
    e = d.c.events[0]
    p = np.load(ROOT / f'data/interim/panel_{e}.npz', allow_pickle=True)
    rows = np.flatnonzero(d.c.row_event == 0)
    y, o = p['y'], p['observed']
    snap_legal = o[:, ::4] & np.isfinite(y[:, ::4])
    assert np.array_equal(d.c.Y[rows][snap_legal], y[:, ::4][snap_legal])
    means = np.nanmean(np.where(o, y, np.nan)[:, :672].reshape(len(rows), 168, 4), axis=2)
    assert np.nanmax(np.abs(means - y[:, :672:4])) > 0          # a mean would differ somewhere


def test_frozen_windows_have_legal_states_and_real_weather(ctx):
    UT, d = ctx
    assert (~d.c.weather_legal).sum() > 0                        # the rule is active
    for s, df in d.df.items():
        assert d.c.weather_legal[df.row.to_numpy()].all()
        assert np.isfinite(d._gather_Y(df, np.arange(-24, 25))).all()


def test_dedup_keeps_the_earliest_event():
    m = pd.read_csv(W / 'locks/US_DEDUP_MAPPING.csv', dtype={'fips': str})
    idx = pd.read_csv(W / 'locks/US_WINDOW_INDEX.csv.gz', dtype={'fips': str})
    kept = set(zip(idx.event, idx.fips, idx.origin_utc))
    assert (m.kept_in_event < m.event).all()
    assert all((k, f, o) in kept for k, f, o in zip(m.kept_in_event, m.fips, m.origin_utc))


def test_batch_alignment_of_history_target_weather_and_clock(ctx):
    UT, d = ctx
    ids = torch.tensor([0, 17, len(d.df['evaluation']) - 1])
    c, y0, step, tgt = d.batch('evaluation', ids)
    for i, wi in enumerate(ids.tolist()):
        r, t = int(d.rows['evaluation'][wi]), int(d.ts['evaluation'][wi])
        assert torch.allclose(c[i, :24] * d.scale, d.Y32[r, t - 24:t], atol=1e-7)
        assert y0[i] == d.Y32[r, t] and torch.equal(tgt[i], d.Y32[r, t + 1:t + 25])
        assert torch.equal(c[i, 24:312].reshape(24, 12), d.Xs[r, t - 23:t + 1])
        assert torch.equal(c[i, 312:600].reshape(24, 12), d.Xs[r, t + 1:t + 25])
        assert torch.equal(step[i, :, :12], d.Xs[r, t + 1:t + 25])
        hrs = d.c.hour_utc[d.c.row_event[r], t + 1:t + 25]
        assert np.allclose(step[i, :, 12].numpy(), np.sin(2 * np.pi * hrs / 24), atol=1e-6)


def test_hierarchical_sampler_is_paired_and_uniform(ctx):
    UT, d = ctx
    a, b = UT.HierSampler(d.df['train'], 8101), UT.HierSampler(d.df['train'], 8101)
    assert np.array_equal(a.draw(512), b.draw(512))
    draws = UT.HierSampler(d.df['train'], 1).draw(400_000)
    ev = d.df['train'].event.to_numpy()[draws]
    freq = pd.Series(ev).value_counts(normalize=True)
    assert len(freq) == 14 and np.abs(freq - 1 / 14).max() < 0.005
    one = d.df['train'].iloc[draws]; one = one[one.event == one.event.iloc[0]]
    cf = one.fips.value_counts(normalize=True)
    assert np.abs(cf - 1 / d.df['train'][d.df['train'].event == one.event.iloc[0]].fips.nunique()).max() < 0.01


def test_evaluation_weights_are_event_equal(ctx):
    UT, d = ctx
    df, w = d.df['evaluation'], d.weights['evaluation']
    per = pd.Series(w).groupby(df.event.to_numpy()).sum()
    assert np.allclose(per.to_numpy(), 1 / 6)


def test_fit_statistics_come_from_train_windows(ctx):
    UT, d = ctx
    df, w = d.df['train'], d.weights['train']
    Ys = d._gather_Y(df, np.arange(-24, 1))
    assert abs(np.sqrt(np.sum(w * np.mean(Ys ** 2, 1))) - d.scale) < 1e-12
