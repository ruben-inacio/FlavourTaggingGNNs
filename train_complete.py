import os
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/$USER/'
import datetime
import subprocess  # to check if running at lipml
host = str(subprocess.check_output(['hostname']))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if "lipml" in host:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "1"  # "0"

import jax
import functools
from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn  
from jax.config import config
from flax.training import train_state, checkpoints
from flax.training.early_stopping import EarlyStopping
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)

import optax       
# from jax_models import TN1, mask_tracks, Predictor
from models.Predictor import Predictor
from models.GN2Plus import TN1
from utils.layers import *

import numpy as np
import argparse
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
import pickle
from train_utils import *



@jax.jit
def train_step(state, batch, key):
    def loss_fn(params):
        out = model.apply(
            {'params': params}, 
            batch['x'], 
            mask, 
            batch['jet_vtx'], 
            batch['trk_vtx'],
            batch['n_tracks'],
            batch['jet_phi'],
            batch['jet_theta'],
        )
        return model.loss(out, batch, mask, mask_edges)
        
    mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])

    # idx = jax.random.permutation(key, 15)
    # batch['x'] = batch['x'][:, idx]
    # batch['trk_y'] = batch['trk_y'][:, idx]
    # batch['edge_y'] = batch['edge_y'].reshape(2500, 15, 15, 2)[:, idx, :, :].reshape(2500, 225, 2)
    # batch['trk_vtx'] = batch['trk_vtx'][:, idx]
    # batch['jet_vtx'] = batch['jet_vtx'][:, idx]
    # # batch['y'] = batch['y'][:, idx]  
    # mask = mask[:, idx]
    # mask_edges = mask_edges[:, idx, :]
    # mask_edges = mask_edges[:, :, idx]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_tasks), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, loss_tasks


@jax.jit
def eval_step(params, batch, key):
    batch_idx = jnp.array(list(range(N_JETS))).repeat(15)

    mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])

    # idx = jax.random.permutation(key, 15)
    # batch['x'] = batch['x'][:, idx]
    # batch['trk_y'] = batch['trk_y'][:, idx]
    # batch['edge_y'] = batch['edge_y'].reshape(2500, 15, 15, 2)[:, idx, :, :].reshape(2500, 225, 2)
    # batch['trk_vtx'] = batch['trk_vtx'][:, idx]
    # batch['jet_vtx'] = batch['jet_vtx'][:, idx]
    # # batch['y'] = batch['y'][:, idx]  
    # mask = mask[:, idx]

    # mask_edges = mask_edges[:, idx, :]
    # mask_edges = mask_edges[:, :, idx]

    out = model.apply(
        {'params': params}, 
        batch['x'], 
        mask, 
        batch['jet_vtx'], 
        batch['trk_vtx'],
        batch['n_tracks'],
        batch['jet_phi'],
        batch['jet_theta'],
    )
    return model.loss(out, batch, mask, mask_edges)


@jax.jit
def test_step(params, batch):
    batch_idx = jnp.array(list(range(N_JETS))).repeat(15)

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


def train_epoch(state, dl, epoch, key, training): 
    running_loss = []
    running_loss_aux = [[], [], [], []]
    dl.pin_memory_device = []

    for i, dd in enumerate(dl):
        for j in range(dd.x.shape[0] // N_JETS):
            x = dd.x[N_JETS*j:N_JETS*(j+1), :, :]
            y = dd.y[N_JETS*j:N_JETS*(j+1), :, :]

            x = jnp.array(x)
            y = jnp.array(y)

            n_jets, n_tracks, _ = x.shape

            batch = get_batch(x, y)

            if training:
                state, loss, loss_tasks = train_step(state, batch, key)
            else:
                loss, loss_tasks = eval_step(state.params, batch, key)
            
            running_loss.append(loss)
            assert(len(loss_tasks) == 4)
            for l in range(len(loss_tasks)):
                running_loss_aux[l].append(loss_tasks[l])

    for l in range(len(running_loss_aux)):
        running_loss_aux[l] = jnp.mean(jnp.array(running_loss_aux[l])).item()
    running_loss = jnp.mean(jnp.array(running_loss)).item()

    if training:
        print('Train - epoch: {}, loss: {}'.format(epoch, running_loss))
    else:
        print('Validation - epoch: {}, loss: {}'.format(epoch, running_loss))

    print("(g, n, e, r) =", tuple(running_loss_aux))
    return state, running_loss, running_loss_aux


def train_model(state, train_dl, valid_dl, save_dir, ensemble_id=0, optimiser='adam', lr=LR_INIT):
    early_stop = EarlyStopping(min_delta=1e-6, patience=20)
    epoch = 0
    ckpt = None
    counter_improvement = 0
    learning_rate = lr
    # TODO need to store losses?
    train_losses = []
    valid_losses = []
    train_losses_aux = []
    valid_losses_aux = []
    import time

    # while epoch < 200:
    while True:
        current_secs = datetime.datetime.now().second
        key_tracks = jax.random.PRNGKey(current_secs)
        t0 = time.time()
        state, train_metrics, train_aux_metrics = train_epoch(state, train_dl, epoch, key_tracks, training=True)
        state, valid_metrics, valid_aux_metrics = train_epoch(state, valid_dl, epoch, key_tracks, training=False)
        t1 = time.time()
        print("TIME = ", t1 - t0)
        train_losses.append(float(train_metrics))
        valid_losses.append(float(valid_metrics))
        train_losses_aux.append(jnp.array(train_aux_metrics, dtype=float).tolist())
        valid_losses_aux.append(jnp.array(valid_aux_metrics, dtype=float).tolist())
        has_improved, early_stop = early_stop.update(valid_metrics)
        if has_improved:
            counter_improvement = 0
            print("Saving model in epoch %d"%epoch)
            ckpt = {
                'epoch': epoch,
                'loss_train': train_metrics,
                'loss_valid': valid_metrics,
                'model': state,
                # 'optimizer': optimizer.state_dict(),
                # 'scalers_features': scalers_features
            }
            checkpoints.save_checkpoint(
                ckpt_dir=save_dir,
                target=ckpt,
                step=ensemble_id,
                overwrite=True,
                keep=100
            )
            with open(save_dir + f"/params_{ensemble_id}.pickle", 'wb') as fpk:
                pickle.dump(state.params, fpk)
        else:
            # elif False:
            counter_improvement += 1
            if counter_improvement == 5:
                learning_rate = learning_rate / 10
                counter_improvement = 0
                state = create_train_state(None, learning_rate=learning_rate, model=model,  params=state.params, optimiser=optimiser)

        if early_stop.should_stop:
            print('Met early stopping criteria, breaking...')
            break

        epoch += 1

    with open(save_dir + f"/loss_history_{ensemble_id}.json", "w") as histf:
        r = json.dumps({
            'train_total': train_losses,
            'valid_total': valid_losses,
            'train_aux': train_losses_aux,
            'valid_aux': valid_losses_aux
        }, indent=4)
        histf.write(r)

    return ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', default=DEFAULT_DIR, help="Directory where the dataset in stored.")
    parser.add_argument('-input_suffix', default=DEFAULT_SUFFIX, help="Name of the dataset to be loaded.")
    parser.add_argument('-save_dir', default=DEFAULT_MODEL_DIR, help="Directory to store results.")
    parser.add_argument('-ensemble_size', default=1, type=int, help="Number of instances to train")
    parser.add_argument('-dev', default=False, type=bool, help="Set to True to check dimensions etc")
    parser.add_argument('-save_plot_data', default=False, type=bool, help="Save final results for plotting?")
    parser.add_argument('-model', type=str)
    parser.add_argument('-name', default="test", type=str)
    parser.add_argument('-optimiser', default="adam", type=str)
    parser.add_argument('-lr', default=LR_INIT, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    opt = parse_args()

    if opt.dev:
        model = get_model(opt.model)
        rng, state = init_model(rng, model)
        print(type(model))
        batch = {
            'x': jnp.ones((N_JETS, 15, 18)), 
            'jet_vtx': jnp.ones((N_JETS,3)), 
            'trk_vtx': jnp.ones((N_JETS, 15, 3)), 
            'jet_y': jnp.ones((N_JETS,3)),
            'trk_y': jnp.ones((N_JETS, 15, 4)),
            'edge_y': jax.nn.one_hot(jnp.ones((N_JETS, 15 ,15)).reshape(-1, 225), 2, axis=2),
            'n_tracks': jnp.ones((N_JETS)),
            'jet_phi': jnp.ones((N_JETS)),
            'jet_theta': jnp.ones((N_JETS))
        }
        k = jax.random.PRNGKey(1)
        with jax.disable_jit():
            print("train begin")
            state, running_loss, aux_losses = train_step(state, batch, k)
            print("train done", len(aux_losses), aux_losses)
            print("eval begin")
            eval_step(state.params, batch, k)
            print("eval done")
            # test_step(state.params, batch)
            # print("test done")
        exit(0)

    print("Loading datasets: start")
    if ".." in opt.input_dir:
        train_dl = torch.load("%s/train_%s.pt"%(opt.input_dir, opt.input_suffix))
        valid_dl = torch.load("%s/valid_%s.pt"%(opt.input_dir, opt.input_suffix))
        test_dl = torch.load("%s/test_%s.pt"%(opt.input_dir, opt.input_suffix))
    else:
        train_dl = torch.load("%s/train_dl.pth"%(opt.input_dir)) # '../training_data/validate_dl.pth'
        valid_dl = torch.load("%s/validate_dl.pth"%(opt.input_dir)) # '../training_data/validate_dl.pth'

        if opt.save_plot_data:
            test_dl = torch.load("%s/test_dl.pth"%(opt.input_dir))
    
    print("Loading datasets: end")
    save_dir = opt.save_dir + "/" + opt.name
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    # save_config_file(save_dir, opt)
    optimiser = opt.optimiser

    model = get_model(opt.model, save_dir=save_dir)
    for instance_id in range(opt.ensemble_size):
        print("Instance number:", instance_id)
        rng, state = init_model(rng, model, optimiser, lr=opt.lr)
        print(type(model), opt.lr)
        ckpt = train_model(state, train_dl, valid_dl, save_dir=save_dir, ensemble_id=instance_id, optimiser=optimiser, lr=opt.lr)
        print(f"Best model stats - epoch {ckpt['epoch']}:")
        print(f"Loss (train, valid) = ({ckpt['loss_train']}, {ckpt['loss_valid']})")
        state = ckpt['model']
        # if opt.save_plot_data:
        #     store_predictions(model, params, test_dl, save_dir, save_truth=False)
            # store_predictions(state.params, test_dl, save_dir, ensemble_id=instance_id)

