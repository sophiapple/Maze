import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from maze.scorers.numpy_energy import EnergyConfig,NumpyEnergy,save_checkpoint,load_checkpoint
from maze.scorers.ensemble import EnergyEnsemble
from maze_paper.scalar_energy import ScalarWeights,score_paths
try:
    import torch
except ImportError:
    torch=None
ROOT=Path(__file__).resolve().parents[1]


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'reproduction/compact/research'/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


class PortableGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(819)
        self.c=EnergyConfig('paper_scalar',8,7,3,5)
        self.x=self.rng.normal(size=(3,6,7,8))
        self.w={k:self.rng.normal(scale=.2,size=s) for k,s in self.c.score_shapes.items()}
        self.indices=np.arange(18).reshape(3,6)
        self.table=self.x.reshape(18,7,8)

    def test_scalar_matches_independent_update_reference(self):
        w=ScalarWeights(self.w['mu.weight'].T,self.w['mass.weight'].T,
                        self.w['fusion.0.weight'].T,self.w['fusion.2.weight'].T)
        np.testing.assert_allclose(NumpyEnergy(self.c,self.w).score(self.table,self.indices),score_paths(self.x,w),atol=1e-12)

    def test_projected_cache_parity(self):
        scorer=NumpyEnergy(self.c,self.w)
        np.testing.assert_array_equal(scorer.score(self.table,self.indices),scorer.score_projected(scorer.project(self.table),self.indices))

    def test_checkpoint_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model';save_checkpoint(path,self.c,self.w,{'role':'test'})
            scorer,_=load_checkpoint(path)
            np.testing.assert_array_equal(scorer.score(self.table,self.indices),NumpyEnergy(self.c,self.w).score(self.table,self.indices))
            with (path/'weights.npz').open('ab') as f:f.write(b'tampered')
            with self.assertRaises(ValueError):load_checkpoint(path)

    def test_no_pickle_or_unknown_parameters(self):
        with self.assertRaises(ValueError):NumpyEnergy(self.c,self.w|{'hidden_gold':np.ones(1)})
        bad=dict(self.w);bad['mu.weight']=np.full((3,8),np.nan)
        with self.assertRaises(ValueError):NumpyEnergy(self.c,bad)

    def test_bad_architecture_rejected(self):
        for kwargs in ({'family':'unknown'},{'views':True},{'shots':0},{'window':-1},{'start_mode':'random'}):
            d={'family':'paper_scalar','dim':8,'views':7};d.update(kwargs)
            with self.assertRaises(ValueError):EnergyConfig(**d)
        with self.assertRaises(ValueError):EnergyConfig('fullqa_vector',8,4)

    def test_consensus_is_exact_weighted_mean(self):
        a=NumpyEnergy(self.c,self.w);b=NumpyEnergy(self.c,{k:v*.5 for k,v in self.w.items()})
        ensemble=EnergyEnsemble([a,b],[1,3])
        np.testing.assert_allclose(ensemble.score(self.table,self.indices),.25*a.score(self.table,self.indices)+.75*b.score(self.table,self.indices))

    def test_invalid_consensus(self):
        a=NumpyEnergy(self.c,self.w)
        for w in ([-1],[0],[float('nan')],[1,2]):
            with self.assertRaises(ValueError):EnergyEnsemble([a],w)
        with self.assertRaises(ValueError):EnergyEnsemble([])

    def test_no_silent_nonfinite_fallback(self):
        bad=self.table.copy();bad[0,0,0]=np.inf
        with self.assertRaises(ValueError):NumpyEnergy(self.c,self.w).score(bad,self.indices)

    def test_path_validation(self):
        scorer=NumpyEnergy(self.c,self.w)
        for idx in (self.indices.astype(float),self.indices[:,:5],np.full_like(self.indices,99)):
            with self.assertRaises(ValueError):scorer.score(self.table,idx)


@unittest.skipIf(torch is None,'Install the train extra to run differentiable and preserved-source parity checks')
class DifferentiableGeometryTests(unittest.TestCase):
    def test_torch_matches_numpy_scalar_and_vector(self):
        from maze.scorers.torch_energy import GeometricEnergy
        for family,views,mode in [('paper_scalar',7,'encoded_system_full_qa'),('compact_vector',4,'zero_start_question_nodes'),('fullqa_vector',7,'encoded_system_full_qa')]:
            c=EnergyConfig(family,8,views,3,5,start_mode=mode)
            torch.manual_seed(19);model=GeometricEnergy(c).double()
            rng=np.random.default_rng(19);x=rng.normal(size=(3,6,views,8))
            if mode=='zero_start_question_nodes':x[:,0]=0
            w={k:v.detach().numpy() for k,v in model.state_dict().items()}
            expected=NumpyEnergy(c,w).score(x.reshape(-1,views,8),np.arange(18).reshape(3,6))
            tensor=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            actual=model(tensor)['logits']
            np.testing.assert_allclose(actual.detach().numpy(),expected,atol=1e-10,rtol=1e-10)
            actual.sum().backward();self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_compact_source_parity(self):
        from maze.scorers.torch_energy import GeometricEnergy
        module=load('maze_replay_geometry','structured_energy_v118.py')
        for mode in ['zero_start_question_nodes','encoded_system_full_qa']:
            torch.manual_seed(23)
            source=module.StructuredEnergyV118(dim=8,views=4,rank=3,hidden=5,start_mode=mode)
            c=EnergyConfig('compact_vector',8,4,3,5,start_mode=mode)
            model=GeometricEnergy(c);model.load_state_dict(source.state_dict())
            x=torch.randn(3,6,4,8)
            if mode=='zero_start_question_nodes':x[:,0]=0
            receipt={'exact_system_input_verified':True,'full_qa_rationale_nodes_verified':True}
            anchors=torch.tensor([0,0,0])
            a=source(x,anchors,system_receipt=receipt);b=model(x,anchors)
            for key in ['logits','energy','static_energy','uplift','harm_probability','log_cost_scaled']:
                torch.testing.assert_close(a[key],b[key],atol=2e-6,rtol=2e-6)
            w={k:v.detach().numpy() for k,v in source.state_dict().items()}
            n=NumpyEnergy(c,w).score(x.numpy().reshape(-1,4,8),np.arange(18).reshape(3,6))
            np.testing.assert_allclose(n,a['logits'].detach().numpy(),atol=2e-5,rtol=2e-5)

    def test_fullqa_source_parity(self):
        from maze.scorers.torch_energy import GeometricEnergy
        load('maze_replay_geometry','structured_energy_v118.py')
        source_module=load('maze_replay_fullqa','structured_energy_v126_portable.py')
        torch.manual_seed(83)
        source=source_module.StructuredEnergyV126(dim=8,views=32,rank=3,hidden=5)
        c=EnergyConfig('fullqa_vector',8,32,3,5)
        model=GeometricEnergy(c);model.load_state_dict(source.state_dict())
        x=torch.randn(2,6,32,8);anchors=torch.tensor([0,0])
        a=source(x,anchors,system_receipt={'exact_system_input_verified':True,'full_qa_rationale_nodes_verified':True})
        b=model(x,anchors)
        for key in ['logits','energy','static_energy','uplift','harm_probability']:
            torch.testing.assert_close(a[key],b[key],atol=2e-5,rtol=2e-5)
