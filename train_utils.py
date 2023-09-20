import os
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/$USER/'
import datetime
import subprocess  # to check if running at lipml
host = str(subprocess.check_output(['hostname']))

import jax.numpy as jnp
import jax
import json
from models.Predictor import Predictor
from models.GN2Plus import TN1
from utils.layers import *
import optax       
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze
import pickle
import copy
import numpy as np

if "lipml" in host:
    DEFAULT_DIR = "/lstore/titan/miochoa/TrackGeometry2023/RachelDatasets_Jun2023/all_flavors/all_flavors" 
else:
    DEFAULT_DIR = '/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/all_flavors'

DEFAULT_SUFFIX =  "alljets_fitting"
DEFAULT_MODEL_DIR = "../models" #"../models" 

# GLOBAL SETTINGS
LR_INIT = 1e-3 #1e-3
N_FEATURES = 18  # DO NOT CHANGE, TO BE REMOVED
N_JETS = 250 
N_TRACKS = 15

def filter_jets(x, y):
    trk_vtx = x[:, :, 16:19]
    jet_vtx = x[:, :, 19:22]
    aligned = jnp.sum(trk_vtx == jet_vtx, axis=2) == 3
    mask = jnp.any(aligned, axis=1) # jnp.repeat(jnp.any(aligned, axis=1)[:, None], x.shape[1], axis=1)#[:, :, None]

    x = x[mask]
    y = y[mask]

    return x, y


def get_batch(x, y):
    # x, y = filter_jets(x, y)

    batch = {}
    batch['x'] = jnp.concatenate((x[:, :, :16], jnp.log(x[:, :, 26:27]), x[:,:, 27:28]),axis=2)
    # batch['x'] = jnp.concatenate((x[:, :, :16], x[:, :, 26:28]),axis=2)
    # ids = jnp.identity(15).reshape(1, 15, 15)
    # ids = ids.repeat(2500, axis=0)
    # batch['x'] = jnp.concatenate([batch['x'], ids], axis=2)
    # Extra args
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


def get_model(model_type, save_dir=None, settings=None):
    if settings is None:
        with open("configs_models.json", "r") as f:
            settings = json.load(f)[model_type]
    # Set random seed
    settings['seed'] = np.random.randint(0, 42)
    if model_type == "predictor":
        model = Predictor(**settings)
    elif model_type == "complete":
        model = TN1(**settings)
    # else:
    #     model = TN1SimpleEnsemble(hidden_channels=32, layers=3, heads=2)
    # model = NDIVE() # 
    if save_dir is not None:
        with open(save_dir + "/config.json", "w") as f:
            json_object = json.dumps(settings, indent=4)
            f.write(json_object)
    return model


def get_init_input():
    x = jnp.ones([N_JETS, N_TRACKS, 51]) * 2.0
    y = jnp.ones([N_JETS, N_TRACKS, 25])
    batch = get_batch(x, y)
    mask, mask_edges = mask_tracks(batch['x'], jnp.ones(x.shape[0]) * 15)

    return batch, mask


def mark_params(params, flag):
    for k, v in params.items():
        if not isinstance(v, dict):
            params[k] = flag
        else:
            params[k] = mark_params(params[k], flag)
    return params


def mask_predictor(params):
    for k in params.keys():
        params[k] =  mark_params(params[k], k == 'apply_strategy_prediction_fn')
        
    return params


def create_train_state(rng, learning_rate, model=None, params=None, optimiser='adam'):
    if params is None:  # Initialise the model
        batch, mask = get_init_input()
        
        params = model.init(rng, 
            batch['x'], 
            mask, 
            batch['jet_vtx'], 
            batch['trk_vtx'],
            batch['n_tracks'],
            batch['jet_phi'],
            batch['jet_theta']
        )['params']
        
        if "Predictor" not in str(type(model)):
            pass
            # print("Loading NDIVE")
            # params = unfreeze(params)
            # with open(f"../models/ndive/params_0.pickle", 'rb') as fp:
            #     ndive_params = pickle.load(fp)
            # params['apply_strategy_prediction_fn'] = ndive_params
            # params = freeze(params)
        else:
            print("Predictor only")
        
    if optimiser == 'adamw':
        tx = optax.adamw(learning_rate=learning_rate)
    elif optimiser == 'adam':
        tx = optax.adam(learning_rate=learning_rate)
    elif optimiser == 'novograd':
        tx = optax.novograd(learning_rate=learning_rate)
    elif optimiser == 'mixed':
        params = unfreeze(params)
        mask_pred = mask_predictor(copy.deepcopy(unfreeze(params)))
        mask_others = jax.tree_map(lambda x: not x, mask_pred)

        tx = optax.chain(
            optax.masked(optax.novograd(learning_rate=learning_rate), mask_pred),
            optax.masked(optax.adamw(learning_rate=learning_rate), mask_others)
        )

    # tx = optax.chain(
    #     optax.adam(learning_rate=learning_rate),
    #     optax.novograd(learning_rate=learning_rate)
    # )
    print("CREATING TRAIN STATE WITH", optimiser, "( lr =", learning_rate, ")")

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def init_model(rng, model, optimiser='adam', lr=LR_INIT):
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, lr, model, optimiser=optimiser)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print("Model and train state created")
    print("Number of parameters:", param_count)
    return rng, state