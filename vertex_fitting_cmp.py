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
    "regression": np.load("../models_vtx_fitting/regression/results_0.npz")['results_graph_reg'],
    "ndive":      np.load("../models_vtx_fitting/ndive/results_0.npz")['results_graph_reg'],
}
true = np.load("../ground_truth_final/jet_vtx.npy")
flavours = np.load("../ground_truth_final/jet_y.npy")
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


true = true.reshape(-1, 3)
for k in preds.keys():
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

# lim1 =  -50
# lim2 =  50
# nbins = 200
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)

# plt.hist(true[:, 0], bins=np.linspace(lim1,lim2,nbins),histtype='step', label="True", lw=3)
# for k in preds.keys():
#     # res = np.sqrt(np.sum(np.square(preds[k] - true), axis=1))
#     plt.hist(preds[k][:, 0], bins=np.linspace(lim1,lim2,nbins),histtype='step', label=labels[k], lw=3)

# ax.set_yscale('log')
# plt.legend()
# plt.savefig("preds_test2.pdf")
# plt.close()


from matplotlib import colors

bjet_color= "white" #"#c7fce7".upper() #"#009988"
cjet_color= "white" #"#faffd1".upper() # "#33BBEE"
ujet_color= "#22dcf5".upper() # "#EE3377"

def make_2dhist_vertex_prediction(x_ax, y_ax, z_ax, outputs, inputs, flavor):
    """ Create 2d histogram of predicted coordinate vs true coordinate for x, y, and z."""

    bins = [50,50]
    range = [[-50, 50], [-50, 50]]
    
    h_x, h_y, h_z = None, None, None
    norm=colors.LogNorm(vmin=10e-6, vmax=1)
    #cmap=colors.LinearSegmentedColormap.from_list("", [ujet_color,cjet_color])
    # cmap='autumn'
    # cmap=colors.LinearSegmentedColormap.from_list("", ['#101f47'.upper(),ujet_color,'whitesmoke'])
    cmap=colors.LinearSegmentedColormap.from_list("", ['#111A44',ujet_color,'whitesmoke'])
    if x_ax is not None: 
        h_x = x_ax.hist2d(inputs[:, 0], outputs[:, 0], bins=bins, range=range, norm=norm, cmap=cmap, density=True)
    if y_ax is not None: 
        h_y = y_ax.hist2d(inputs[:, 1], output[:, 1], bins=bins, range=range, norm=norm, cmap=cmap, density=True)
    if z_ax is not None: 
        h_z = z_ax.hist2d(inputs[:, 2], outputs[:, 2], bins=bins, range=range, norm=norm, cmap=cmap, density=True)

    for ax, dim in zip([x_ax, y_ax, z_ax], ['X', 'Y', 'Z']):
        if ax is None: continue
        ax.set_xlabel(f'${dim}_{{true}}$ [mm]')#, fontsize = 14)
        ax.set_ylabel(f'${dim}_{{pred}}$ [mm]')#, fontsize = 14)

    return h_x, h_y, h_z


# Filter
cond = flavours < 2
true = true[cond]
for k in preds:
    preds[k] = preds[k][cond]

fig, axs = plt.subplots(2, 2, figsize=(12,10), dpi=100, constrained_layout=True)
hxb, _, hzb = make_2dhist_vertex_prediction(axs[0][0], None, axs[0][1], true, preds['ndive'],"n")
hxc, _, hzc = make_2dhist_vertex_prediction(axs[1][0], None, axs[1][1], true, preds['regression'],"r")
axs[0][0].text(-48, 40, 'NDIVE')
axs[0][1].text(-48, 40, 'NDIVE')
axs[1][0].text(-48, 40, 'Regression')
axs[1][1].text(-48, 40, 'Regression')
axs[0][0].set_facecolor(bjet_color)
axs[0][0].patch.set_alpha(0.25)
axs[0][1].set_facecolor(bjet_color)
axs[0][1].patch.set_alpha(0.25)
axs[1][0].set_facecolor(cjet_color)
axs[1][0].patch.set_alpha(0.25)
axs[1][1].set_facecolor(cjet_color)
axs[1][1].patch.set_alpha(0.25)
axs[0][0].set_xticks([-50,0,50])
axs[0][1].set_xticks([-50,0,50])
axs[1][0].set_xticks([-50,0,50])
axs[1][1].set_xticks([-50,0,50])
axs[0][0].set_yticks([-50,0,50])
axs[0][1].set_yticks([-50,0,50])
axs[1][0].set_yticks([-50,0,50])
axs[1][1].set_yticks([-50,0,50])
fig.colorbar(hxb[3], ax=axs.ravel().tolist())
#fig.colorbar(hxb[3], ax=axs[0][1])
#fig.colorbar(hxc[3], ax=axs[1][1])
#fig.tight_layout()
plt.savefig("ndive_vs_regression_3.pdf")