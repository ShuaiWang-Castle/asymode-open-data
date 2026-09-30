"""D09 scheduling limits and exclusive source registration checks."""
from pathlib import Path
import sys
import json
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import d09_queue as Q


@pytest.mark.parametrize('workers',[1,2])
def test_worker_limit_allowed(tmp_path,workers):
    p=tmp_path/'WORKERS';p.write_text(str(workers)+'\n')
    assert Q.worker_count(p)==workers


@pytest.mark.parametrize('workers',[0,3,4,-1])
def test_worker_limit_rejects_expansion(tmp_path,workers):
    p=tmp_path/'WORKERS';p.write_text(str(workers)+'\n')
    with pytest.raises(ValueError):Q.worker_count(p)


def test_finite_queue_has_exact_six_distinct_jobs():
    assert len(Q.JOBS)==len(set(Q.JOBS))==6
    assert {arm for arm,fold in Q.JOBS}=={'host','crk','new'}
    assert {fold for arm,fold in Q.JOBS}=={2,3}


def test_atomic_status_leaves_readable_complete_receipt(tmp_path):
    p=tmp_path/'RUN_STATUS.json'
    Q.atomic_status(p,{'status':'training','active':{'new_f2':123}})
    Q.atomic_status(p,{'status':'training_complete','active':{}})
    assert json.loads(p.read_text())=={'status':'training_complete','active':{}}
    assert not p.with_suffix('.next').exists()


def test_registration_rejects_changed_scientific_source(tmp_path,monkeypatch):
    p=tmp_path/'write.py';p.write_text('changed\n')
    monkeypatch.setattr(Q,'SOURCES',[p]);monkeypatch.setattr(Q,'ROOT',tmp_path)
    def fake(cmd,**kwargs):
        return 'a'*40+'\n' if '--verify' in cmd else b'original\n'
    monkeypatch.setattr(Q.subprocess,'check_output',fake)
    with pytest.raises(RuntimeError,match='mismatch'):Q.registered('a'*40)


def preflight_fixture(tmp_path,monkeypatch,drop=1.):
    monkeypatch.setattr(Q,'ROOT',tmp_path)
    source=tmp_path/'write.py';source.write_text('registered\n')
    resource=dict(schema='d09_fit_only_resource_v1',passed=True,
        arm='new',heldout_fold=3,registration_commit='a'*40,
        heldout_scores_computed=False,resumable_model_saved=False,
        forced_initial_engine_step=200,forced_initial_alpha=.1,
        updates=[dict(finite_gradients=True,drop_mask=drop,loss=.1,
            kernel_parameters_missing_gradients=[],optimizer_covers_all_trainable=True,
            engine_step=201+i) for i in range(3)])
    rp=tmp_path/'PREFLIGHT_RESOURCE.json';rp.write_text(json.dumps(resource))
    done=dict(schema='d09_preflight_done_v1',preflight=True,arm='new',heldout_fold=3,
        steps=3,registration_commit='a'*40,source_sha256={'write.py':Q.sha(source)},
        outputs={'PREFLIGHT_RESOURCE.json':Q.sha(rp)})
    path=tmp_path/'PREFLIGHT_DONE.json';path.write_text(json.dumps(done))
    return path


def test_open_kernel_preflight_receipt_validated(tmp_path,monkeypatch):
    path=preflight_fixture(tmp_path,monkeypatch)
    assert Q.preflight_receipt(path,'new','a'*40)['forced_initial_alpha']==.1


def test_closed_drop_preflight_rejected(tmp_path,monkeypatch):
    path=preflight_fixture(tmp_path,monkeypatch,drop=0.)
    with pytest.raises(RuntimeError,match='open-kernel'):Q.preflight_receipt(path,'new','a'*40)


def test_preflight_changed_source_or_saved_checkpoint_rejected(tmp_path,monkeypatch):
    path=preflight_fixture(tmp_path,monkeypatch)
    source=tmp_path/'write.py';source.write_text('changed\n')
    with pytest.raises(RuntimeError,match='source changed'):Q.preflight_receipt(path,'new','a'*40)
    source.write_text('registered\n');(tmp_path/'final.pt').write_bytes(b'unexpected')
    with pytest.raises(RuntimeError,match='saved model'):Q.preflight_receipt(path,'new','a'*40)
