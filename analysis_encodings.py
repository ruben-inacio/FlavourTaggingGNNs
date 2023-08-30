
import os
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/$USER/'
import datetime
import subprocess  # to check if running at lipml
host = str(subprocess.check_output(['hostname']))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if "lipml" in host:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "1"  # "0"

import json
import pickle
from models.GN2Plus import TN1
import jax.numpy as jnp
import jax
import datetime
import matplotlib.pyplot as plt
from flax.core import freeze, unfreeze
import torch
from train_utils import get_batch
from utils.layers import mask_tracks

base_path = "../models_v2/gn2ndive_atlas"
model_path = "../models_v2/gn2ndive_atlas_onehot_none"


with open(base_path + "/config.json", "r") as f:
    base_settings = json.load(f)

with open(model_path + "/config.json", "r") as f:
    model_settings = json.load(f) 

with open(base_path + "/params_0.pickle", 'rb') as f:
    base_params = pickle.load(f)

with open(model_path + "/params_0.pickle", 'rb') as f:
    model_params = pickle.load(f)

base = TN1(**base_settings)
model = TN1(**model_settings)

if False:
    for name in base_params['preprocessor']['track_init']:
        fig, ax = plt.subplots(1, 2,figsize=(16,8))

        print(name, 
            base_params['preprocessor']['track_init'][name]['kernel'].shape, 
            model_params['preprocessor']['track_init'][name]['kernel'].shape)
        data_base = base_params['preprocessor']['track_init'][name]['kernel']
        data_model = model_params['preprocessor']['track_init'][name]['kernel']
        vmin = min(data_base.min(), data_model.min())
        # print(data_model[5, 1], data_model[6, 1])
        vmax = max(data_base.max(), data_model.max())
        cmap = "winter"
        d1 = ax[0].imshow(data_base, cmap=cmap, vmin=vmin, vmax=vmax)
        d2 = ax[1].imshow(data_model, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        fig.colorbar(d1)
        fig.colorbar(d2)
        plt.savefig(f"analysis/weights_{name}.png")

# for k in range(10):
#     os.makedirs("analysis/jet{:02d}".format(k))

# exit(0)

N_JETS = 250
test_dl = torch.load("/lstore/titan/miochoa/TrackGeometry2023/RachelDatasets_Jun2023/all_flavors/all_flavors/test_dl.pth")
for i, dd in enumerate(test_dl):
    if i == 1:
        break
    for j in range(dd.x.shape[0] // N_JETS):
        if j == 1:
            break

        x = dd.x[N_JETS*j:N_JETS*(j+1), :, :]
        y = dd.y[N_JETS*j:N_JETS*(j+1), :, :]

        x = jnp.array(x)
        y = jnp.array(y)

        n_jets, n_tracks, _ = x.shape

        batch = get_batch(x, y)
        mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])
        out = model.apply(
            {'params': model_params}, 
            batch['x'], 
            mask, 
            batch['jet_vtx'], 
            batch['trk_vtx'],
            batch['n_tracks'],
            batch['jet_phi'],
            batch['jet_theta'],
        )

        probs_model, reprs_model = out[0], out[-1]
        
        out = base.apply(
            {'params': base_params}, 
            batch['x'], 
            mask, 
            batch['jet_vtx'], 
            batch['trk_vtx'],
            batch['n_tracks'],
            batch['jet_phi'],
            batch['jet_theta'],
        )
        probs_base, reprs_base = out[0], out[-1]

        for k in range(10):
            true = jnp.argmax(batch['jet_y'][k, :])
            model_pred = probs_model[k, :]
            base_pred = probs_base[k, :]
            with open("analysis/jet{:02d}/predictions.txt".format(k), "w") as fres:
                fres.write("true = " + str(true) + '\n')
                fres.write("base =" + str(base_pred) +'\n')
                fres.write("model =" + str(model_pred) + '\n')
            
            fig, ax = plt.subplots(1, 2,figsize=(10,5))
            vmin = min(reprs_base[k, :].min(), reprs_model[k, :].min())
            vmax = max(reprs_base[k, :].max(), reprs_model[k, :].max())
            cmap = "winter"
            d1 = ax[0].imshow(reprs_base[k, :], cmap=cmap, vmin=vmin, vmax=vmax)
            d2 = ax[1].imshow(reprs_model[k, :], cmap=cmap, vmin=vmin, vmax=vmax)
            plt.tight_layout()
            fig.colorbar(d1)
            fig.colorbar(d2)
            plt.savefig("analysis/jet{:02d}/representations.pdf".format(k))
            plt.close()
        