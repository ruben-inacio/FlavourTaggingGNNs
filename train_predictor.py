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
import jax.numpy as jnp     
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)

import flax
from flax import linen as nn          
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze, unfreeze
import optax       

import numpy as np
import json

import datetime
import argparse
import importlib
from models.Predictor import Predictor
from utils.layers import mask_tracks
from flax.training.early_stopping import EarlyStopping
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pickle
from functools import partial


def parse_args():
    """
    Argument parser for training script.
    """
    parser = argparse.ArgumentParser(description="Train the model.")

    parser.add_argument(
        "-s",
        "--samples",
        default= "/lstore/titan/miochoa/TrackGeometry2023/RachelDatasets_Jun2023/all_flavors/all_flavors",
        type=str,
        help='Path to training samples.'
    )
    
    parser.add_argument(
        "-n",
        "--num_gpus",
        default=1,
        type=int,
        help='Number of GPUs to use.'
    )
    
    parser.add_argument(
        "-b",
        "--batch_size",
        default=100,
        type=int,
        help='Number of jets per batch.'
    )
    
    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help='Number of epochs for training.'
    )
    
    parser.add_argument(
        "-l",
        "--learning_rate",
        default=1e-5,
        type=float,
        help='Learning rate for training.'
    )
    
    parser.add_argument(
        "-p",
        "--pretrained_NDIVE",
        default=False,
        type=bool,
        help='Train the larger ftag model with pretrained NDIVE.'
    )
    
    parser.add_argument(
        "-c",
        "--cont",
        default=False,
        type=bool,
        help='Continue training a pretrained model.'
    )
    
    args = parser.parse_args()
    return args

def main():
    def get_batch(x, y):
        batch = {}
        batch['x'] = jnp.concatenate((x[:, :, :16], x[:, :, 26:28]),axis=2)

        batch['n_tracks'] = x[:, 0, 22]
        batch['jet_phi'] = x[:, 0, 24]
        batch['jet_theta'] = x[:, 0, 25]

        batch['jet_y'] = y[:, 0, 18:21]
        batch['trk_y'] = y[:, :, 21:25]
        batch['edge_y'] = jax.nn.one_hot(y[:, :, 3:18].reshape(-1, 225), 2, axis=2)
        batch['trk_vtx'] = x[:, :, 16:19]
        batch['jet_vtx'] = x[:, 0, 19:22]
        batch['y'] = y  
        return batch
    # parse args
    args = parse_args()
    
    train_dl = torch.load('{}/train_dl.pth'.format(args.samples))
    validate_dl = torch.load('{}/validate_dl.pth'.format(args.samples))
    print("Data loaded")    

    nominal_batch_size = 10000

    DEVICE_COUNT = jax.device_count()

    TRAIN_VMAP_COUNT = int(nominal_batch_size/DEVICE_COUNT/args.batch_size)
    TEST_VMAP_COUNT = int(nominal_batch_size/DEVICE_COUNT/args.batch_size)
    
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    num_epochs = args.epochs
    with open("configs_models.json", "r") as f:
        settings = json.load(f)["predictor"]
    model = Predictor(**settings)
    params = model.init(
        init_rng, 
        jnp.ones([10,15,51]),
        jnp.ones([10, 15, 1]).astype(bool), 
        jnp.ones([10,3]), 
        jnp.ones([10,15,3]), 
        jnp.ones([10]), 
        jnp.ones([10]), 
        jnp.ones([10])
    )['params']

    #total_steps = num_epochs*len(train_dl)*DEVICE_COUNT*TRAIN_VMAP_COUNT
    #cosine_decay_schedule = optax.cosine_decay_schedule(
    #    args.learning_rate, decay_steps=total_steps, alpha=0.5
    #)
    optimizer = optax.chain(
        #optax.novograd(learning_rate=cosine_decay_schedule)
        optax.novograd(learning_rate=args.learning_rate)
    )
    tx = optimizer
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
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

    def train_epoch(key, state, train_ds, epoch):

        state_dist = flax.jax_utils.replicate(state)

        running_loss = []
        running_loss_aux = [[], [], [], []]

        for i, d in enumerate(train_ds):

            if i%10==0: print('Batch #{}'.format(i))
            #if i>=1: break

            x = jnp.array(d.x, dtype=jnp.float64)[:,:,0:30]
            y = jnp.array(d.y, dtype=jnp.float64)

            # shuffle jets in each batch during training
            j,k,l = x.shape
            idx = jax.random.permutation(key, j)
            x = x[idx]
            y = y[idx]

            x_batch = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TRAIN_VMAP_COUNT, -1, *m.shape[1:])), x)
            y_batch = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TRAIN_VMAP_COUNT, -1, *m.shape[1:])), y)    

            #with jax.disable_jit():
            loss, loss_tasks, grads = train_step_pmap(key, state_dist, x_batch, y_batch)   

            state = flax.jax_utils.unreplicate(state_dist)
            state = update_model(state, grads)
            state_dist = flax.jax_utils.replicate(state)
            
            running_loss.append(loss)
            assert(len(loss_tasks) == 4)
            for l in range(len(loss_tasks)):
                running_loss_aux[l].append(loss_tasks[l])

        for l in range(len(running_loss_aux)):
            running_loss_aux[l] = jnp.mean(jnp.array(running_loss_aux[l])).item()
        running_loss = jnp.mean(jnp.array(running_loss)).item()

        print('Training - epoch: {}, loss: {}'.format(epoch, running_loss))
        print("(g, n, e, r) =", tuple(running_loss_aux))

        return state, running_loss, running_loss_aux

    def eval_model(key, state, test_ds, epoch):

        state_dist = flax.jax_utils.replicate(state)

        running_loss = []
        running_loss_aux = [[], [], [], []]

        for i, d in enumerate(test_ds):

            if i%10==0: print('Batch #{}'.format(i))
            #if i>=1: break

            x = jnp.array(d.x, dtype=jnp.float64)[:,:,0:30]
            y = jnp.array(d.y, dtype=jnp.float64)

            # shuffle jets in each batch during training
            j,k,l = x.shape
            idx = jax.random.permutation(key, j)
            x = x[idx]
            y = y[idx]

            x_batch = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TEST_VMAP_COUNT, -1, *m.shape[1:])), x)
            y_batch = jax.tree_map(lambda m: m.reshape((DEVICE_COUNT, TEST_VMAP_COUNT, -1, *m.shape[1:])), y)

            #with jax.disable_jit():
            loss, loss_tasks = eval_step_pmap(key, state_dist, x_batch, y_batch)

            running_loss.append(loss)
            assert(len(loss_tasks) == 4)
            for l in range(len(loss_tasks)):
                running_loss_aux[l].append(loss_tasks[l])

        for l in range(len(running_loss_aux)):
            running_loss_aux[l] = jnp.mean(jnp.array(running_loss_aux[l])).item()
        running_loss = jnp.mean(jnp.array(running_loss)).item()

        print('Validation - epoch: {}, loss: {}'.format(epoch, running_loss))
        print("(g, n, e, r) =", tuple(running_loss_aux))
        return running_loss, running_loss_aux
    
    early_stop = EarlyStopping(min_delta=1e-6, patience=20)
    epoch = 0
    ckpt = None
    counter_improvement = 0
    learning_rate = args.learning_rate
    # TODO need to store losses?
    train_losses = []
    valid_losses = []
    train_losses_aux = []
    valid_losses_aux = []
    save_dir = "../models/ndive"
    import time
    while True:
        key = jax.random.PRNGKey(datetime.datetime.now().second)
        t0 = time.time()
        state, train_metrics, train_aux_metrics = train_epoch(key, state, train_dl, epoch)
        valid_metrics, valid_aux_metrics = eval_model(key, state, validate_dl, epoch)
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
                step=0,
                overwrite=True,
                keep=100
            )
            with open(save_dir + f"/params_0.pickle", 'wb') as fpk:
                pickle.dump(state.params, fpk)

        if early_stop.should_stop:
            print('Met early stopping criteria, breaking...')
            break

        epoch += 1

    with open(save_dir + f"/loss_history_0.json", "w") as histf:
        r = json.dumps({
            'train_total': train_losses,
            'valid_total': valid_losses,
            'train_aux': train_losses_aux,
            'valid_aux': valid_losses_aux
        }, indent=4)
        histf.write(r)
        
if __name__ == "__main__":
    main()

