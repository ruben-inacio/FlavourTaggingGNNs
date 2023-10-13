import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from train_utils import DEFAULT_DIR, get_batch
import os
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/$USER/'
import datetime
import subprocess  # to check if running at lipml
host = str(subprocess.check_output(['hostname']))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if "lipml" in host:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "1"  # "0"

import pickle
import jax
import jax.numpy as jnp
import json
from models.Predictor import Predictor
from utils.layers import mask_tracks
import utils.data_format as daf

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
test_dl = torch.load("%s/test_dl.pth"%(DEFAULT_DIR))
batch_size = 250

preds = {
    "relativity": [],
    "regression": [],
    "ndive":      [],
   # "ndive_perf": []
}
true = []

def relativity(x, mask, true_jet, true_trk, n_tracks, jet_phi, jet_theta):
        jet_pt = x[:, 0, daf.JetData.TRACK_JET_PT]
        jet_eta = x[:, 0, daf.JetData.TRACK_JET_ETA]

        # start by calculating absolute displacement
        jet_theta = 2*jnp.arctan(jnp.exp(-jet_eta))
        jet_mom = jet_pt*jnp.cosh(jet_eta)
        b_mom = 0.70 * jet_mom # on average, 70% of the b-quark energy goes into the b-hadron
        tau = 1.55e-12 # s
        mass = 5 # GeV
        c = 3e8 # m/s
        gamma = jnp.sqrt(1+jnp.square(b_mom/mass))
        d = gamma * tau * c

        # turn that into a vertex estimate, assuming jet axis direction
        x = d * jnp.cos(jet_phi) * 1e3 # mm
        y = d * jnp.sin(jet_phi) * 1e3 # mm
        z = d * jnp.cos(jet_theta) * 1e3 # mm

        points = jnp.stack([x, y, z], axis=1)
        log_errors = jnp.zeros([x.shape[0], 3])

        return points


with open(f"../models_ndive_section_potential/regression/params_0.pickle", 'rb') as fp:
    params_reg = pickle.load(fp)
with open(f"../models_ndive_section_potential/regression/config_0.json", 'r') as fp:
    settings = json.load(fp)
regression = Predictor(**settings)

with open(f"../models_ndive_section_potential/ndive/params_0.pickle", 'rb') as fp:
    params_ndive = pickle.load(fp)
with open(f"../models_ndive_section_potential/ndive/config_0.json", 'r') as fp:
    settings = json.load(fp)
ndive = Predictor(**settings)

with open(f"../models_ndive_section_potential/ndive_perfect/config_0.json", 'r') as fp:
    settings = json.load(fp)
ndive_perf = Predictor(**settings)

def test_step(params, model, batch):
    mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])
    return model.apply(
        {'params': params}, 
        batch['x'], 
        mask, 
        batch['jet_vtx'], 
        batch['trk_vtx'],
        batch['n_tracks'],
        batch['jet_phi'],
        batch['jet_theta'],
    )[3]

for i, dd in enumerate(test_dl):
    for j in range(dd.x.shape[0] // batch_size):
        x = dd.x[batch_size*j:batch_size*(j+1), :, :]
        y = dd.y[batch_size*j:batch_size*(j+1), :, :]

        x = jnp.array(x)
        y = jnp.array(y)

        n_jets, n_tracks, _ = x.shape

        batch = get_batch(x, y, 0)
        mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])

        true.append(batch['jet_vtx'])
        preds['relativity'].append(relativity(batch['x'], mask, batch['jet_vtx'], batch['trk_vtx'], batch['n_tracks'], batch['jet_phi'], batch['jet_theta']))
        preds['regression'].append( test_step(params_reg, regression, batch))
        preds['ndive'].append(test_step(params_ndive, ndive, batch))
        #preds['ndive_perf'].append(test_step(params_ndive, ndive_perf, batch))




true = jnp.array(true)
true = true.reshape(-1, 3)
for k in preds.keys():
    preds[k] = jnp.array(preds[k])
    preds[k] = preds[k].reshape(-1, 3)

labels = {
    "regression": "Graph Regression",
    "relativity": "Relativity-based",
    "ndive":  "NDIVE",
    # "ndive_perf": "NDIVE (perfect)"
}

font = {'family' : 'sans-serif',
        'size'   : 22}
plt.rc('font', **font)

lim1 = -50
lim2 =  50
nbins = 200
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
plt.hist(true[:, 0], bins=np.linspace(lim1,lim2,nbins),histtype='step', label="True", lw=3)
for k in preds.keys():
    plt.hist(preds[k][:, 0], bins=np.linspace(lim1,lim2,nbins),histtype='step', label=labels[k], lw=3)

ax.set_yscale('log')
plt.legend()
plt.savefig("preds_test.pdf")
plt.close()