"""Synthetic tests for the D09 train worker; no real data/training jobs launched."""
import copy
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

import d09_train as T

torch.set_num_threads(2)


def fixture(n=40):
    rng=np.random.default_rng(209)
    return dict(xu=rng.normal(size=(n,216,14)).astype(np.float32),
                xr=rng.normal(size=(n,216,20)).astype(np.float32),
                xo=rng.normal(size=(n,216,4)).astype(np.float32),
                geo=rng.normal(size=(n,40)).astype(np.float32),
                y=rng.uniform(.01,.15,size=(n,144)).astype(np.float32),
                m=np.ones((n,144),np.float32),y0=rng.uniform(0,.03,size=n).astype(np.float32),
                fips=np.asarray([f'c{i:04d}' for i in range(n)]),
                regime=np.asarray([T.REGIMES[i%5] for i in range(n)]),w=rng.uniform(.5,3,size=n))


class StreamsAndSplits(unittest.TestCase):
    def test_C_and_Fortran_1D_2D_3D_exact_selected_rows(self):
        for shape in ((8,),(8,4),(8,3,4)):
            for order in ('C','F'):
                data=np.arange(np.prod(shape),dtype=np.float32).reshape(shape,order=order)
                data=np.array(data,order=order)
                with tempfile.TemporaryDirectory() as name:
                    p=Path(name)/'a.npz';np.savez_compressed(p,x=data)
                    with zipfile.ZipFile(p) as archive:actual=T.stream_subset(archive,'x',np.array([0,2,7]),n=8)
                np.testing.assert_array_equal(actual,data[[0,2,7]])

    def test_unsorted_repeated_indices_rejected(self):
        for ids in ([2,1],[1,1]):
            with self.assertRaises(ValueError):T.stream_subset(None,'x',np.asarray(ids),n=8)

    def test_fold_masks_disjoint_and_fullheld_contains_unsampled_units(self):
        data=dict(fold=np.array([2,2,3,3,4,4]),panel=np.array([1,0,1,0,1,1],bool),
                  group=np.array(['a','a','b','b','c','c']))
        idx=T.job_indices(data,2)
        np.testing.assert_array_equal(idx['fit'],[2,4,5])
        np.testing.assert_array_equal(idx['held_panel'],[0])
        np.testing.assert_array_equal(idx['held_full'],[0,1])

    def test_merged_group_leakage_rejected(self):
        data=dict(fold=np.array([2,3]),panel=np.ones(2,bool),group=np.array(['a','a']))
        with self.assertRaisesRegex(ValueError,'leakage'):T.job_indices(data,2)

    def test_unregistered_fold_rejected(self):
        with self.assertRaises(ValueError):T.job_indices({},4)


class TrainingObjective(unittest.TestCase):
    def test_weights_match_original_formula_with_innerFIT_only(self):
        F=fixture(10);fit=np.array([0,1,2,3,4]);out,info=T.design_weighted(F,fit)
        raw=np.zeros(10)
        for r in set(F['regime'][fit]):
            ii=fit[F['regime'][fit]==r];zero=np.sum(F['w'][ii,None]*F['m'][ii]*F['y'][ii].astype(float)**2)
            raw[ii]=F['w'][ii]/zero
            self.assertAlmostEqual(info['Z_regime'][r],zero)
        scale=F['m'][fit].sum()/np.sum(F['m'][fit]*raw[fit,None])
        np.testing.assert_array_equal(out['m_train'],(F['m']*raw[:,None]*scale).astype(np.float32))
        self.assertTrue(np.all(out['m_train'][5:]==0))

    def test_heldout_labels_and_weights_do_not_change_innerFIT_target(self):
        F=fixture(10);other=copy.deepcopy(F);fit=np.arange(5)
        other['y'][5:]=1;other['w'][5:]=1e9
        a,ai=T.design_weighted(F,fit);b,bi=T.design_weighted(other,fit)
        np.testing.assert_array_equal(a['m_train'][fit],b['m_train'][fit]);self.assertEqual(ai,bi)

    def test_zero_target_risk_rejected(self):
        F=fixture(10);F['y'][:]=0
        with self.assertRaises(ValueError):T.design_weighted(F,np.arange(5))

    def test_heldout_covariates_do_not_change_model_statistics(self):
        F=fixture(10);other=copy.deepcopy(F);fit=np.arange(5)
        for k in ('xu','xr','xo','geo'):other[k][5:]+=10000
        a=T.G.fit_stats(F,fit);b=T.G.fit_stats(other,fit)
        for k in a:
            for x,y in zip(a[k],b[k]):np.testing.assert_array_equal(x,y)

    def test_fullFIT_chunk_gradient_accumulation_matches_complete_update(self):
        class Small(torch.nn.Module):
            kernel=None;haz_beta=None
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(3,1)
            def forward(self,b):return dict(P=torch.sigmoid(self.linear(b['xu']).squeeze(-1)))
        torch.manual_seed(22);model=Small();reference=copy.deepcopy(model)
        fit=dict(xu=torch.randn(11,144,3),y=torch.rand(11,144),m=torch.rand(11,144)+.1)
        def engine(micro,m):
            e=object.__new__(T.G.Engine);e.model=m;e.fit=fit;e.fit_idx=np.arange(11);e.microbatch_size=micro
            e.opt=torch.optim.Adam(m.parameters(),lr=.003);e.step=0;e.arm='W+Cin';e.last_loss=float('nan');e.training_trace=[]
            return e
        full,chunk=engine(None,reference),engine(3,model)
        x,y=full.train_step(),chunk.train_step()
        self.assertAlmostEqual(x,y,places=6)
        for a,b in zip(reference.parameters(),model.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


class FreshInitialization(unittest.TestCase):
    def test_host_old_new_shared_initialization_and_new_parameters_fresh(self):
        F=fixture();fit=np.arange(40);weighted,_=T.design_weighted(F,fit)
        models=[T.Engine(weighted,fit,arm) for arm in T.ARMS]
        self.assertEqual(len({e.initial_host_parameter_sha256 for e in models}),1)
        self.assertEqual(models[1].initial_kernel_parameter_sha256,models[2].initial_kernel_parameter_sha256)
        self.assertTrue(all(e.microbatch_size==512 for e in models))
        self.assertEqual(models[1].model.kernel.last_calibration['n_counties'],40)
        self.assertEqual(models[2].model.kernel.last_calibration['n_counties'],40)
        for e in models:
            self.assertEqual([g['lr'] for g in e.opt.param_groups],[.003,.0003])
            ids={id(p) for g in e.opt.param_groups for p in g['params']}
            self.assertEqual(ids,{id(p) for p in e.model.parameters()})

    def test_synthetic_step0_prediction_all_arms_recover_host(self):
        F=fixture();fit=np.arange(40);weighted,_=T.design_weighted(F,fit)
        predictions=[]
        for arm in T.ARMS:
            e=T.Engine(weighted,fit,arm);e.model.eval()
            with torch.no_grad():predictions.append(e.model(T.G.make_batch(F,np.array([0,1]),e.stats))['P'])
        for prediction in predictions[1:]:torch.testing.assert_close(prediction,predictions[0],rtol=0,atol=0)


class ScoringAndProtocol(unittest.TestCase):
    def test_score_perfect_prediction_and_false_peak_count(self):
        F=fixture(5);F['y'][:]=.01;F['y'][0,:]=.2
        perfect=T.score(F,F['y'].copy(),np.arange(5))
        self.assertEqual(perfect['all']['design_RMSE'],0);self.assertEqual(perfect['S']['peak_ratio_design_weight_median'],1)
        p=F['y'].copy();p[1,10]=.2
        self.assertEqual(T.score(F,p,np.arange(5))['nonS']['severe_alarms']['n'],1)

    def test_scores_observed_hours_only(self):
        F=fixture(5);F['y'][:]=0;F['m'][:,10]=0
        p=F['y'].copy();p[:,10]=1
        score=T.score(F,p,np.arange(5))
        self.assertEqual(score['all']['design_SSE'],0);self.assertEqual(score['nonS']['severe_alarms']['n'],0)

    def test_fixed_budget_no_early_stop_or_HT_optimization(self):
        self.assertEqual(T.STEPS,900);self.assertEqual(T.SEED,0);self.assertEqual(T.CHECKPOINTS,(0,100,300,900))
        self.assertEqual(T.PREFLIGHT_STEPS,3);self.assertFalse(T.PROTOCOL['early_stopping'])
        self.assertIn('not HT',T.PROTOCOL['training_weights'])

    def test_status_atomic_rewrite_own_file(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'RUN_STATUS.json';T.status(path,phase='training',step=1);T.status(path,phase='completed',step=2)
            import json
            self.assertEqual(json.loads(path.read_text())['step'],2)
            self.assertFalse(path.with_name(path.name+'.tmp').exists())


if __name__=='__main__':unittest.main()
