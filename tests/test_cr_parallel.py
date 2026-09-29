"""Synthetic checks for compute-only I20 handoff; no panel or live jobs."""
import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/geo_weather_20260924'))
import run_cr_parallel as P


def identities():
    parent = dict(pid=100, ppid=1, started='Tue Sep 29 00:05:00 2026',
                  command=f'python {P.R.HERE}/run_cr_screen.py --execute --primary-cohort S')
    child = dict(pid=101, ppid=100, started='Tue Sep 29 00:05:01 2026',
                 command=f'python {P.R.HERE}/screen.py --label v1_crk_s0 --arm CRK+Cin '
                         '--data v1D --folds 1 --steps 900 --seed 0 --threads 2')
    status = dict(status='training', runner_pid=100, active={'1': 101})
    return status, parent, child


def test_adoption_checks_identity_command_ownership_and_config():
    status, parent, child = identities()
    P.check_adoption(status, parent, child, 100, 1, 101)
    for bad in [dict(child, ppid=999), dict(child, pid=999),
                dict(child, command=child['command'].replace('--seed 0', '--seed 1')),
                dict(child, command=child['command'].replace('--threads 2', '--threads 4'))]:
        with pytest.raises(RuntimeError):
            P.check_adoption(status, parent, bad, 100, 1, 101)
    with pytest.raises(RuntimeError):
        P.check_adoption(dict(status, active={'1': 101, '2': 102}), parent, child, 100, 1, 101)
    with pytest.raises(RuntimeError):
        P.check_adoption(status, dict(parent, command='python unrelated.py'), child, 100, 1, 101)


def test_only_verified_coordinator_receives_signal(monkeypatch):
    _, parent, child = identities()
    sent = []
    monkeypatch.setattr(P, 'process_identity', lambda pid: parent)
    monkeypatch.setattr(P.os, 'kill', lambda pid, sig: sent.append((pid, sig)))
    P.terminate_coordinator(parent)
    assert sent == [(parent['pid'], signal.SIGTERM)]
    assert all(pid != child['pid'] for pid, _ in sent)
    monkeypatch.setattr(P, 'process_identity', lambda pid: dict(parent, started='different process'))
    with pytest.raises(RuntimeError, match='identity changed'):
        P.terminate_coordinator(parent)
    assert len(sent) == 1


def test_adopted_child_can_be_reparented_but_pid_reuse_is_rejected():
    _, _, child = identities()
    assert P.same_process(child, dict(child, ppid=1))
    assert not P.same_process(child, dict(child, started='different process'))
    assert not P.same_process(child, None)


def test_pending_never_duplicates_owned_or_done_folds():
    assert P.pending_folds([], {1: {}, 2: {}, 3: {}}) == [4, 5]
    assert P.pending_folds([1, 3], {2: {}}) == [4, 5]
    assert P.pending_folds([1, 2, 3, 4, 5], {}) == []
    with pytest.raises(RuntimeError):
        P.pending_folds([1], {1: {}})
    with pytest.raises(RuntimeError):
        P.pending_folds([], {6: {}})


def test_dynamic_workers_bounded_to_authorized_maximum(tmp_path):
    path = tmp_path / 'PARALLEL_WORKERS'
    for n in (1, 2, 3):
        path.write_text(str(n))
        assert P.worker_count(path) == n
    for bad in ('0', '4', '-1', '2.5', 'bad'):
        path.write_text(bad)
        with pytest.raises(ValueError):
            P.worker_count(path)


def test_process_parser_handles_dead_reparented_and_reused_process(monkeypatch):
    class Response:
        returncode = 0
        stdout = '101 1 RN Tue Sep 29 00:05:01 2026 python /public/experiment/screen.py --seed 0\n'
        def check_returncode(self):
            assert self.returncode == 0
    response = Response()
    monkeypatch.setattr(P.subprocess, 'run', lambda *a, **kw: response)
    result = P.process_identity(101)
    assert result['pid'] == 101 and result['ppid'] == 1
    assert result['started'] == 'Tue Sep 29 00:05:01 2026'
    assert result['command'] == 'python /public/experiment/screen.py --seed 0'
    response.returncode, response.stdout = 1, ''
    assert P.process_identity(101) is None


def test_default_plan_does_not_inspect_or_signal_processes(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['run_cr_parallel.py'])
    monkeypatch.setattr(P, 'process_identity', lambda *a: pytest.fail('plan touched a process'))
    P.main()
    assert json.loads(capsys.readouterr().out) == dict(train=False, workers=3, threads=2, preserve_live_training=True)
