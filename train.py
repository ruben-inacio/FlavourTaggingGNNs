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
import flax
from jax.config import config
from flax.training import train_state, checkpoints
from flax.training.early_stopping import EarlyStopping
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)

import optax       
from functools import partial
# from jax_models import TN1, mask_tracks, Predictor
from models.Predictor import Predictor
from models.GN2Plus import TN1
from models.GN2Simple import TN1SimpleEnsemble
from utils.layers import *

import numpy as np
import argparse
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
import pickle
from train_utils import *
import time


@partial(jax.pmap, axis_name="device", in_axes=(None, 0, 0, 0),  out_axes=(0, 0, 0))
def train_step_pmap(key, state, batch_x, batch_y):  

    def train_step(state, batch_x, batch_y):
        def loss_fn(params):
            batch = get_batch(batch_x, batch_y)
            mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])
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
            loss, losses = model.loss(out, batch, mask, mask_edges)
            return loss, losses
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, losses), grads = grad_fn(state.params)
        return loss, losses, grads

    train_step_vmap = jax.vmap(train_step, in_axes=(None, 0, 0), out_axes=(0, 0, 0))

    loss, losses, grads = train_step_vmap(state, batch_x, batch_y)
    loss_total = jnp.mean(loss)

    return loss_total, losses, grads


@partial(jax.pmap, axis_name="device", in_axes=(None, 0, 0, 0),  out_axes=(0, 0))
def eval_step_pmap(key, state, batch_x, batch_y):    

    def eval_step(state, batch_x, batch_y):
        batch = get_batch(batch_x, batch_y)
        mask, mask_edges = mask_tracks(batch['x'], batch['n_tracks'])
        out = model.apply(
                {'params': state.params}, 
                batch['x'], 
                mask, 
                batch['jet_vtx'], 
                batch['trk_vtx'],
                batch['n_tracks'],
                batch['jet_phi'],
                batch['jet_theta'],
            )
        loss, losses = model.loss(out, batch, mask, mask_edges)
        return loss, losses

    eval_step_vmap = jax.vmap(eval_step, in_axes=(None, 0, 0), out_axes=(0, 0))

    loss, losses = eval_step_vmap(state, batch_x, batch_y)
    loss_total = jnp.mean(loss)

    return loss_total, losses


@jax.jit
def update_model(state, grads):

    def gradient_application(i, st):

        m = jnp.int32(jnp.floor(i/TRAIN_VMAP_COUNT))
        n = jnp.int32(jnp.mod(i,TRAIN_VMAP_COUNT))
        grad = jax.tree_util.tree_map(lambda x: x[m][n], grads)
        st = st.apply_gradients(grads=grad)
        return (st)

    state = jax.lax.fori_loop(0, DEVICE_COUNT*TRAIN_VMAP_COUNT, gradient_application, state)

    return state 


def train_epoch(state, dl, epoch, key, training): 
    state_dist = flax.jax_utils.replicate(state)

    running_loss = []
    running_loss_aux = [[], [], [], []]

    for i, d in enumerate(dl):

        x = jnp.array(d.x, dtype=jnp.float64)
        y = jnp.array(d.y, dtype=jnp.float64)


        x = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TRAIN_VMAP_COUNT, -1, *m.shape[1:])), x)
        y = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TRAIN_VMAP_COUNT, -1, *m.shape[1:])), y)
        
        if training:
            loss, loss_tasks, grads = train_step_pmap(key, state_dist, x, y)
            state = flax.jax_utils.unreplicate(state_dist)
            state = update_model(state, grads)
            state_dist = flax.jax_utils.replicate(state)
        else:
            loss, loss_tasks = eval_step_pmap(key, state_dist, x, y)

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
    if training:
        return state, running_loss, running_loss_aux
    else:
        return running_loss, running_loss_aux


def train_model(state, train_dl, valid_dl, save_dir, ensemble_id=0):
    early_stop = EarlyStopping(min_delta=1e-6, patience=20)
    epoch = 0
    ckpt = None
    counter_improvement = 0
    learning_rate = LR_INIT
    # TODO need to store losses?
    train_losses = []
    valid_losses = []
    train_losses_aux = []
    valid_losses_aux = []

    # while epoch < 200:
    while True:
        current_secs = datetime.datetime.now().second
        key = jax.random.PRNGKey(current_secs)
        t0 = time.time()
        state, train_metrics, train_aux_metrics = train_epoch(state, train_dl, epoch, key, training=True)
        valid_metrics, valid_aux_metrics = train_epoch(state, valid_dl, epoch, key, training=False)
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
                state = create_train_state(None, learning_rate=learning_rate, model=model, params=state.params)

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
    parser.add_argument('-batch_size', default=2500, type=int, help="Batch size")
    parser.add_argument('-ensemble_size', default=1, type=int, help="Number of instances to train")
    parser.add_argument('-dev', default=False, type=bool, help="Set to True to check dimensions etc")
    parser.add_argument('-save_plot_data', default=False, type=bool, help="Save final results for plotting?")
    parser.add_argument('-model', type=str)
    parser.add_argument('-name', default="test", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    opt = parse_args()


    nominal_batch_size = 10000

    DEVICE_COUNT = jax.device_count()

    TRAIN_VMAP_COUNT = int(nominal_batch_size/DEVICE_COUNT/opt.batch_size)
    TEST_VMAP_COUNT = int(nominal_batch_size/DEVICE_COUNT/opt.batch_size)

    if opt.dev:
        model = get_model(opt.model)
        rng, state = init_model(rng, model)
        print(type(model))
        state_dist = flax.jax_utils.replicate(state)
        x = jnp.ones([N_JETS, 15, 51])
        y = jnp.ones([N_JETS, 15, 30])
        x = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TRAIN_VMAP_COUNT, -1, *m.shape[1:])), x)
        y = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TRAIN_VMAP_COUNT, -1, *m.shape[1:])), y)
            
        k = jax.random.PRNGKey(1)
        # with jax.disable_jit():
        print("train begin")
        train_step_pmap(k, state_dist, x, y)
        print("train done")
        print("eval begin")
        eval_step_pmap(k, state_dist, x, y)
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

    model = get_model(opt.model, save_dir=save_dir)
    for instance_id in range(opt.ensemble_size):
        print("Instance number:", instance_id)
        rng, state = init_model(rng, model)
        print(type(model))
        ckpt = train_model(state, train_dl, valid_dl, save_dir=save_dir, ensemble_id=instance_id)
        print(f"Best model stats - epoch {ckpt['epoch']}:")
        print(f"Loss (train, valid) = ({ckpt['loss_train']}, {ckpt['loss_valid']})")
        state = ckpt['model']
        # if opt.save_plot_data:
        #     store_predictions(model, params, test_dl, save_dir, save_truth=False)
            # store_predictions(state.params, test_dl, save_dir, ensemble_id=instance_id)

