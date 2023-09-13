import sys
sys.path.append("../")
import os
import subprocess  # to check if running at lipml
host = str(subprocess.check_output(['hostname']))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if "lipml" in host:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from flax.training import checkpoints
import jax
import torch
import pickle
from models.Predictor import Predictor
from models.GN2Plus import TN1
from utils.layers import mask_tracks
import numpy as np
import argparse

from train_utils import get_batch, DEFAULT_MODEL_DIR, DEFAULT_DIR, get_model
import json
import random
# Configuring seeds
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# params = checkpoints.restore_checkpoint(ckpt_dir='rachel', target=None, step=0, parallel=False)['params']
# model = Network()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', default=DEFAULT_DIR, help="Directory where the dataset in stored.")
    parser.add_argument('-save_dir', default=DEFAULT_MODEL_DIR, help="Directory to store results.")
    parser.add_argument('-name', default="test", type=str)
    parser.add_argument('-model', type=str)
    parser.add_argument('-truth_only', default=False, type=bool)
    return parser.parse_args()


N_JETS = 250

@jax.jit
def test_step(params, batch):
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
    )[:6]

def store_predictions(model, params, dl, save_dir, scy=None, save_truth=False, ensemble_id=0, truth_only=False):
    pred_mu = []
    pred_sigma = []
    true = []
    true_flavours = []
    jet_pts = []
    jet_etas = []
    jet_trks = []

    true_nodes = []
    pred_nodes = []
    true_edges = []
    pred_edges = []
    pred_flavours = []
    out_graph, out_nodes, out_edges, p_mu, p_var = [None] * 5
    dl.pin_memory_device = []
    for i, dd in enumerate(dl):
        print("batch ", i, "/", len(dl))
        # if i == 1:
        #     break
        for j in range(dd.x.shape[0] // N_JETS):
            x = dd.x[N_JETS*j:N_JETS*(j+1), :, :]
            y = dd.y[N_JETS*j:N_JETS*(j+1), :, :]

            x = np.array(x)
            y = np.array(y)
        
            jet_pts.append(x[:, 0, 26])
            jet_etas.append(x[:, 0, 27])
            jet_trks.append(x[:, 0, 22])
            batch = get_batch(x, y)
            
            mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])
            mask = mask[:, :, 0]
            mask_edges = mask_edges.reshape(-1, 225)
            
            trk_y = np.argmax(batch['trk_y'], axis=2)
            trk_y = np.where(mask, trk_y, -1).reshape(-1) #trk_y[mask].reshape(-1)
            true_nodes.append(trk_y)

            edge_y = np.argmax(batch['edge_y'], axis=2)
            edge_y = np.where(mask_edges, edge_y, -1).reshape(-1) # edge_y[mask_edges].reshape(-1)
            true_edges.append(edge_y)
    
            if not truth_only:
                out_graph, out_nodes, out_edges, p_mu, p_var, _ = test_step(params, batch)
                
                if out_nodes is not None:
                    pred_nodes.append(out_nodes)#[mask])
                if out_edges is not None:
                    pred_edges.append(out_edges)#[mask_edges])
                if out_graph is not None:
                    pred_flavours.append(out_graph)
        
                this_pred_mu, this_pred_var = p_mu, p_var 

                if this_pred_mu is not None:
                    if scy is not None:
                        this_pred_mu = np.array(scy.inverse_transform(this_pred_mu))
                    pred_mu.append(this_pred_mu)
                
                if this_pred_var is not None:
                    this_pred_sigma = this_pred_var
                    # this_pred_sigma = np.sqrt(np.exp(this_pred_var))
                    if scy is not None:
                        this_pred_sigma = np.array(scy.inverse_transform(this_pred_sigma.reshape(-1,3)))
                    pred_sigma.append(this_pred_sigma)
            
            this_true = batch['jet_vtx'] # d.jet_vtx.reshape(-1, 3)
            if scy is not None:
                this_true = np.array(scy.inverse_transform(this_true))
            true.append(this_true)

            true_flavours.append(np.argmax(batch['jet_y'], axis=1))
    ####

    true = np.concatenate(true)
    true_flavours = np.concatenate(true_flavours)
    jet_pts = np.concatenate(jet_pts)
    jet_etas = np.concatenate(jet_etas)
    jet_trks = np.concatenate(jet_trks)
    
    true_nodes = np.concatenate(true_nodes)
    true_edges = np.concatenate(true_edges)

    if pred_nodes != []:
        pred_nodes = np.concatenate(pred_nodes)
        print(pred_nodes.shape)
        np.save(f'{save_dir}/results_nodes_clf_{ensemble_id}.npy', pred_nodes)
    if pred_edges != []:
        pred_edges = np.concatenate(pred_edges)
        np.save(f'{save_dir}/results_edges_clf_{ensemble_id}.npy', pred_edges)
        print(pred_edges.shape)
    if pred_flavours != []:
        pred_flavours = np.concatenate(pred_flavours)
        print(pred_flavours.shape)
        np.save(f'{save_dir}/results_graph_clf_{ensemble_id}.npy', pred_flavours)
    if pred_mu != []:
        pred_mu = np.concatenate(pred_mu)
        print(pred_mu.shape)
        np.save(f'{save_dir}/results_graph_reg_{ensemble_id}.npy', pred_mu)
    if pred_sigma != []:
        pred_sigma = np.concatenate(pred_sigma)
        np.save(f'{save_dir}/results_graph_reg_var_{ensemble_id}.npy', pred_sigma)
        print(pred_sigma.shape)

    print(true.shape)
    print(true_flavours.shape)
    print(jet_pts.shape)
    print(jet_etas.shape)
    print(jet_trks.shape)
    print(true_edges.shape)
    print(true_nodes.shape)
    print("valid_nodes =", (true_nodes > -1).sum())
    print("valid_edges =", (true_edges > -1).sum())
    if save_truth:
        np.save('../ground_truth/jet_vtx.npy', true)
        np.save('../ground_truth/trk_y.npy', true_nodes)
        np.save('../ground_truth/edge_y.npy', true_edges)
        np.save('../ground_truth/jet_y.npy', true_flavours)
        np.save('../ground_truth/jet_pts.npy', jet_pts)
        np.save('../ground_truth/jet_etas.npy', jet_etas)
        np.save('../ground_truth/jet_trks.npy', jet_trks)


    return pred_mu, pred_sigma, true, true_flavours, jet_pts, jet_trks   


if __name__ == '__main__':
    opt = parse_args()

    test_dl = torch.load("%s/test_dl.pth"%(opt.input_dir))

    if opt.truth_only:
        store_predictions(None, None, test_dl, None, save_truth=True, truth_only=True)
        exit(0)

    save_dir = opt.save_dir + "/" + opt.name

    if not os.path.exists(save_dir): 
        raise ValueError("Path does not exist.")
    else:
        print("Retrieving from", save_dir)

    with open(f"{save_dir}/config.json", "r") as f:
        settings = json.load(f)

    try:
        model = TN1(**settings)
    except Exception:
        model = Predictor(**settings)

    total = len([filename for filename in os.listdir(save_dir) if filename.startswith("params")])
    print("model", type(model), "num instances", total)
    start = 0
    for i in range(total):
        results_exist = sum(x.endswith(f'{i}.npy') for x in os.listdir(save_dir)) > 0
        if results_exist:
            start = i+1
        else:
            break
            

    for ensemble_id in range(start, total):
        with open(f"{save_dir}/params_{ensemble_id}.pickle", 'rb') as fp:
            params = pickle.load(fp)

        # store_predictions(model, params, test_dl, save_dir, save_truth=ensemble_id == 0, ensemble_id=ensemble_id)
        store_predictions(model, params, test_dl, save_dir, save_truth=False, ensemble_id=ensemble_id)
        # Restoring seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
