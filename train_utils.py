import jax.numpy as jnp
import jax
import json
from models.Predictor import Predictor
from models.GN2Plus import TN1
from models.GN2Simple import TN1SimpleEnsemble
from utils.layers import *
import optax       
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze
import pickle

DEFAULT_DIR = "/lstore/titan/miochoa/TrackGeometry2023/RachelDatasets_Jun2023/all_flavors/all_flavors" 
DEFAULT_SUFFIX =  "alljets_fitting"
DEFAULT_MODEL_DIR = "../models_v2" #"../models" 

# GLOBAL SETTINGS
LR_INIT = 1e-3
N_FEATURES = 18  # DO NOT CHANGE, TO BE REMOVED
N_JETS = 2500 #0 
N_TRACKS = 15

def get_batch(x, y):
    # c_jets = y[:, 0, 19] == 1
    # y = y[c_jets]
    # x = x[c_jets]

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


def create_train_state(rng, learning_rate, model=None, params=None):
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
            print("Loading NDIVE")
            params = unfreeze(params)
            with open(f"../models/ndive/params_0.pickle", 'rb') as fp:
                ndive_params = pickle.load(fp)
            params['apply_strategy_prediction_fn'] = ndive_params
            params = freeze(params)
        else:
            print("Predictor only")
    tx = optax.adam(learning_rate=learning_rate)
    # tx = optax.novograd(learning_rate=learning_rate)
    print("CREATING TRAIN STATE WITH LR =", learning_rate)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def init_model(rng, model):
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, LR_INIT, model)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print("Model and train state created")
    print("Number of parameters:", param_count)
    return rng, state