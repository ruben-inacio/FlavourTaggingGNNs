
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
import numpy as np
import jax
import datetime
import matplotlib.pyplot as plt
from flax.core import freeze, unfreeze
import torch
from train_utils import get_batch
from utils.layers import mask_tracks
import random
# Configuring seeds
seed = 1906
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

path = "../models_slac/gn2_eye"

with open(path + "/params_0.pickle", 'rb') as f:
    params = pickle.load(f)





for layer in range(3):
    l = f"enc_layer_{layer}"
    for k in list(params['preprocessor']['encoder'][l]['attn'].keys()):
        print(k, params['preprocessor']['encoder'][l]['attn'][k]['kernel'].shape)
        print(k, params['preprocessor']['encoder'][l]['attn'][k]['bias'].shape)
        fig, ax = plt.subplots(1, 2,figsize=(16,8))

        kernel = params['preprocessor']['encoder'][l]['attn'][k]['kernel'][:, 0, :]
        bias = params['preprocessor']['encoder'][l]['attn'][k]['kernel'][:, 1, :]
        # bias = params['preprocessor']['encoder'][l]['attn'][k]['bias']
        if bias.ndim == 1:
            bias = bias.reshape(1, bias.shape[0])
        vmin = min(kernel.min(), bias.min())
        # print(data_model[5, 1], data_model[6, 1])
        vmax = max(kernel.max(), bias.max())
        cmap = "winter"
        d1 = ax[0].imshow(kernel, cmap=cmap, vmin=vmin, vmax=vmax)
        d2 = ax[1].imshow(bias, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        fig.colorbar(d1)
        fig.colorbar(d2)
        plt.savefig(f"../analysis/layer_{layer}/attn_weights_{k}_{path.split('/')[-1]}.png")
        plt.close()

print(list(params['preprocessor']['encoder']['enc_layer_0']['lin1'].keys()))
# print(list(params['preprocessor']['encoder'].keys()))


exit(0)
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

if True:
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
        plt.savefig(f"../analysis/mlps/weights_{name}.png")

# for k in range(10):
#     os.makedirs("analysis/jet{:02d}".format(k))

# exit(0)

N_JETS = 250
test_dl = torch.load("/lstore/titan/miochoa/TrackGeometry2023/RachelDatasets_Jun2023/all_flavors/all_flavors/test_dl.pth")
print("dataset loaded")

summary = [["Quality level", "Distance", "Prediction", "True"]]

for i, dd in enumerate(test_dl):
    # if i == 1:
    #     break
    for j in range(dd.x.shape[0] // N_JETS):
        # if j == 1:
        #     break

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
        probs_base, pred_vtx, reprs_base = out[0], out[3], out[-1]
        for k in range(N_JETS):

            true = jnp.argmax(batch['jet_y'][k, :])
            model_pred = probs_model[k, :]
            base_pred = probs_base[k, :]

            # Quality level assignment
            performance_diff = model_pred[true] - base_pred[true]
            if performance_diff < -.4:
                lvl = 1
            elif performance_diff < -.2:
                lvl = 2
            elif performance_diff < .1:
                lvl = 3
            elif performance_diff < .2:
                lvl = 4
            else:
                lvl = 5
            

            # Distance category assignment for summary 
            distance = jnp.sqrt(jnp.sum(jnp.square(pred_vtx[k, :] - batch['jet_vtx'][k, :])))
            if distance < jnp.sqrt(.01*3):
                category = "Very close"
            if distance < jnp.sqrt(.25*3):
                category = "Close"
            elif distance < 2:
                category = "Below 2"
            else:
                category = "Far"

            pred_dist = jnp.sqrt(jnp.sum(jnp.square(pred_vtx[k, :] - jnp.array([0,0,0]))))
            true_dist = jnp.sqrt(jnp.sum(jnp.square(batch['jet_vtx'][k, :] - jnp.array([0,0,0]))))
            if pred_dist < jnp.sqrt(.01*3):
                category_pred = "Very close to 000"
            if pred_dist < jnp.sqrt(.25*3):
                category_pred = "Close to 000"
            elif pred_dist < 2:
                category_pred = "Not that close"
            else:
                category_pred = "Far"

            if true_dist < jnp.sqrt(.01*3):
                category_true = "Very close to 000"
            if true_dist < jnp.sqrt(.25*3):
                category_true = "Close to 000"
            elif true_dist < 2:
                category_true = "Not that close"
            else:
                category_true = "Far"

            summary.append([lvl, category, category_pred, category_true])
            print(summary[-1])
            continue  #FIXME remove to store jet data
            if not os.path.exists("../analysis/lvl{}/jet{:02d}".format(lvl, k)):
                os.makedirs("../analysis/lvl{}/jet{:02d}".format(lvl, k))
            with open("../analysis/lvl{}/jet{:02d}/predictions.txt".format(lvl, k), "w") as fres:
                fres.write("true = " + str(true) + '\n')
                fres.write("true vtx = " + str(batch['jet_vtx'][k, :]) + '\n')
                fres.write("pred vtx = " + str(pred_vtx[k, :]) + '\n')
                fres.write("base =" + str(base_pred) +'\n')
                fres.write("model =" + str(model_pred) + '\n')
            
            fig, ax = plt.subplots(1, 2,figsize=(10,5))
            vmin = min(reprs_base[k, :].min(), reprs_model[k, :].min())
            vmax = max(reprs_base[k, :].max(), reprs_model[k, :].max())
            cmap = "winter"
            d1 = ax[0].imshow(reprs_base[k, :], cmap=cmap, vmin=vmin, vmax=vmax)
            d2 = ax[1].imshow(reprs_model[k, :], cmap=cmap, vmin=vmin, vmax=vmax)
            # d3 = ax[1, 0].imshow(reprs_model[k, :32], cmap=cmap, vmin=vmin, vmax=vmax)
            # d4 = ax[1, 1].imshow(reprs_model[k, 32:], cmap=cmap, vmin=vmin, vmax=vmax)
            plt.tight_layout()
            fig.colorbar(d1)
            fig.colorbar(d2)
            plt.savefig("../analysis/lvl{}/jet{:02d}/representations.pdf".format(lvl, k))
            plt.close()
        
summary = [summary[0]] + sorted(summary[1:], key = lambda x: x[0])
lvl45 = list(filter(lambda x: x[0] > 3, summary[1:]))
lvl123 = list(filter(lambda x: x[0] < 4, summary[1:]))
with open("../analysis/summary_all.txt", "w") as fsummary:
    for line in range(len(summary)):
        fsummary.write("{},{:20s},{:20s},{:20s},\n".format(*summary[line]))
        