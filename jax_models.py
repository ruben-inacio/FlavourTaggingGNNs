import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn  
from jax.config import config
import functools
from typing import (Any, Callable, Optional, Tuple)
from flax.linen.dtypes import promote_dtype
from sklearn.preprocessing import StandardScaler
import copy
import pickle

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
import datetime
from flax import linen as nn          
from flax.training import train_state
import optax       
from data_utils import custom_linspace
import numpy as np
from endtoend import Extrapolator

from jax_fit import ndive
from jax_extrapolation import extrapolation


def binary_cross_entropy(ytrue, ypred, weights):    
    loss = -ytrue * jnp.log(ypred) - (1. - ytrue) * jnp.log(1-ypred)
    
    if weights != None:
        loss = loss * weights
    
    return loss
    

def categorical_cross_entropy(ytrue, ypred, weights):    
    loss = -ytrue * jnp.log(ypred)
    
    if weights != None:
        loss = loss * weights
    
    return jnp.sum(loss, axis=2)


def squared_error(targets, predictions, mean=True):
    errors = (predictions - targets) ** 2
    if mean:
        return errors.mean()
    return errors

class GlobalAttention(nn.Module):
    gate_nn: nn.Module
    
    def setup(self):
        self.softmax = lambda x: nn.activation.softmax(x, axis=1)
    
    # def __call__(self, x, batch_size, mask=None):
    def __call__(self, x, mask=None):
        batch_size, length, _ = x.shape

        gate = self.gate_nn(x)

        # length = gate.shape[0] // batch_size
        # gate = gate.reshape(batch_size, length, 1)
        # x = x.reshape(batch_size, length, x.shape[-1])

        if mask is not None:
            # if mask.ndim == 1:
            #     mask = mask.reshape(mask.shape[0], 1)

            # mask = mask.reshape(batch_size, length, 1)
            gate = jnp.where(mask, gate, jnp.array([-jnp.inf]))

        gate = self.softmax(gate)
        out = jnp.sum(x * gate, axis=1)

        return out, gate


class EncoderLayer(nn.Module):
    heads: int
    hidden_channels: int
    architecture:  str

    def setup(self):
        self.weights = None
        self.attn = nn.SelfAttention(num_heads=self.heads)
        self.norm1 = nn.LayerNorm()
        self.lin1 = nn.Dense(features=4*self.hidden_channels)

        self.norm2 = nn.LayerNorm()
        self.lin2 = nn.Dense(features=self.hidden_channels)
        self.fwd_fn = eval("self.fwd_" + self.architecture)
    
    def fwd_post(self, x):
        n = self.attn(x)
        x = x + n
        x = self.norm1(x)
        n = self.lin1(x)
        n = nn.relu(n)
        n = self.lin2(n)
        x = x + n
        x = self.norm2(x)
        return x

    def fwd_pre(self, x):
        n = self.norm1(x)
        n = self.attn(n)
        x = x + n
        n = self.norm2(x)
        n = self.lin1(n)
        n = nn.relu(n)
        n = self.lin2(n)
        x = x + n
        return x

    def fwd_postb2t(self, x):
        pass

    # def __call__(self, x, batch_size, mask=None):
    def __call__(self, x, mask=None):
        # x = x.reshape(batch_size, x.shape[0] // batch_size, x.shape[-1]) 
        x = self.fwd_fn(x)
        # x = x.reshape(-1, x.shape[-1])
        return x


class Encoder(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    architecture: str

    def setup(self):
        for i in range(self.layers):
            setattr(self, f"enc_layer_{i}", EncoderLayer(heads=self.heads, hidden_channels=self.hidden_channels, architecture=self.architecture))
   
        if self.architecture == "pre":
            self.norm3 = nn.LayerNorm()
        else:
            self.norm3 = lambda x: x

    def __call__(self, g, mask=None):
        for i in range(self.layers):
            g = getattr(self, f"enc_layer_{i}")(g, mask=mask)

        if mask is not None:
            g = g * mask
        
        # g.reshape(batch_size, g.shape[0] // batch_size, g.shape[-1]) 
        g = self.norm3(g)
        if mask is not None:
            g = g * mask
        # g = g.reshape(-1, g.shape[-1])
        return g
        
        
# Vertexing Modules

class PreProcessor(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    architecture: str

    def setup(self):
        # TODO insert trak init #layers as class param 
        self.track_init = nn.Sequential([
            nn.Dense(features=self.hidden_channels),
            nn.relu,
            nn.Dense(features=self.hidden_channels),
            nn.relu,
            nn.Dense(features=self.hidden_channels),
            nn.relu,
            nn.Dense(features=self.hidden_channels),
        ])
        
        self.encoder = Encoder(hidden_channels=self.hidden_channels, heads=self.heads, layers=self.layers, architecture=self.architecture)
        
    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones((x.shape[0], 1))

        # mask = mask.reshape([-1, 1])
        x = x * mask

        t = self.track_init(x)
        t = t * mask
        # TODO bnorm

        g = self.encoder(t, mask=mask)
        g = g * mask

        return t, g


class Regression(nn.Module):
    hidden_channels: int

    def setup(self):
        self.gate_nn = nn.Dense(features=1)
        self.pool = GlobalAttention(gate_nn=self.gate_nn)
        
        self.mlp_mean = nn.Sequential([
            nn.Dense(self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels),
            nn.relu,
            nn.Dense(3)
        ])
            
        self.mlp_var = nn.Sequential([
            nn.Dense(self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels),
            nn.relu,
            nn.Dense(3)
        ])

    def __call__(self, x, repr_track, weights, mask, *args):
        pooled, _ = self.pool(weights * repr_track, mask=mask)
        out_mean = self.mlp_mean(pooled)
        out_var = self.mlp_var(pooled)

        return out_mean, out_var, None


class NDIVEv2(nn.Module):
    def setup(self):
        pass

    def __call__(self, *args):
        pass


class Predictor(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    strategy_sampling: str
    method: str
    n_tracks = 15

    def setup(self):
        self.preprocessor = PreProcessor(
            hidden_channels = self.hidden_channels,
            layers = self.layers,
            heads = self.heads,
            architecture="post"
        )

        if self.strategy_sampling == "compute":
            self.nn_sample = nn.Sequential([
                nn.Dense(features=self.hidden_channels),
                nn.relu,
                nn.Dense(features=1)
            ])

        self.apply_strategy_sampling_fn = eval("self.apply_strategy_sampling_" + self.strategy_sampling)

        if self.method == "regression":
            self.fitting_method = Regression(hidden_channels=self.hidden_channels)
        else:
            self.fitting_method = lambda x, _, wv, mask: ndive(x, jnp.where(mask.squeeze(), wv.squeeze(), 1e-100), jnp.zeros([x.shape[0], 3]))

    def apply_strategy_sampling_none(self, repr_track, mask, true_jet, true_trk):
        weights = jnp.ones(mask.shape)
        weights = (weights * mask)
        return weights

    def apply_strategy_sampling_perfect(self, repr_track, mask, true_jet, true_trk):
        true_trk = true_trk.reshape(-1 ,3)
        true_jet = true_jet.repeat(self.n_tracks, axis=0)
        weights = (true_jet[:, 0] == true_trk[:, 0]) & (true_jet[:, 1] == true_trk[:, 1]) & (true_jet[:, 2] == true_trk[:, 2])
        weights = weights.reshape(*mask.shape)

        weights = (mask & weights).astype(float)
        return weights

    def apply_strategy_sampling_compute(self, repr_track, mask, true_jet, true_trk):
        def sample(w):
            current_date = datetime.datetime.now()
            key = jax.random.PRNGKey(current_date.second)
            selection = jax.random.bernoulli(key, w) # jnp.sqrt(w))
            w = selection * w
            w = nn.activation.softmax(w, axis=1)
            return w

        weights = self.nn_sample(repr_track)
        weights = jnp.where(mask, weights, jnp.array([-jnp.inf]))
        weights = nn.activation.softmax(weights, axis=1)
        weights = jax.lax.stop_gradient(sample(weights))

        return weights

    def __call__(self, x, mask, true_jet, true_trk, *args):
        # dummy = jnp.mean(x, where=mask, axis=1, keepdims=True).repeat(self.n_tracks, axis=1)
        # x = jnp.where(mask, x, dummy)
        # mask = jnp.ones(mask.shape)
        # if x.shape[2] > 16:
        #     x = x[:, :, :16]
        t, g = self.preprocessor(x, mask)

        repr_track = jnp.concatenate([t, g], axis=2)

        weights = self.apply_strategy_sampling_fn(repr_track, mask, true_jet, true_trk)            
        out_mean, out_var, out_chi = self.fitting_method(x, repr_track, weights, mask)
        
        out_mean = jnp.clip(out_mean, a_min=-4000., a_max=4000.)
        out_mean = jax.numpy.nan_to_num(out_mean, nan=4000., posinf=4000., neginf=-4000.)
        out_var = jax.numpy.nan_to_num(out_var, nan=1000., posinf=1000., neginf=1000.)

        if out_chi is not None:
            out_var = jax.numpy.nan_to_num(out_var, nan=1000., posinf=1000., neginf=1000.)

        return None, None, None, out_mean, out_var, out_chi

    def loss(self, out, batch, mask, mask_edges):
        _, _, _, out_mean, out_var, out_chi = out

        loss_f = jnp.sqrt(jnp.sum((batch['jet_vtx']-out_mean)**2, axis=1))
        loss_f = jnp.mean(loss_f) #, where=batch['jet_y'][:, 0] != 1)
        
        # loss_mse = squared_error(batch['jet_vtx'], out_mean)

        # loss_f = jnp.sqrt(jnp.sum((batch['jet_vtx']-out_mean)**2, axis=1))
        weights = (batch['x'][:, 0, 16])
        loss_f = jnp.average(loss_f, weights=weights)

        loss = loss_f # + .2 * loss_mse

        return loss, (0, 0, 0, loss_f)


class TN1(nn.Module):
    hidden_channels :    int
    layers:              int
    heads:               int
    augment:             bool
    strategy_prediction: str
    points_as_features:  bool  # FIXME outdated
    scale:               bool
    one_hot:             bool
    
    def debug_print(*args):
        for arg in args:
            print("[DEBUG]", arg)

    def setup(self):
        assert((self.augment and self.strategy_prediction is not None) or not self.augment)
        self.preprocessor = PreProcessor(
            hidden_channels = self.hidden_channels,
            layers = self.layers,
            heads = self.heads,
            architecture="post"
        )
        
        self.gate_nn = nn.Dense(features=1)
        self.pool = GlobalAttention(gate_nn=self.gate_nn)

        self.extraplator = None
        if self.augment:
            self.extrapolator = extrapolation
        
        if self.scale:
            if self.points_as_features:
                self.scaler=pickle.load(open("../training_data/scaler_30jun.npy",'rb'))
            else:
                self.scaler=pickle.load(open("../training_data/scaler_28jun.npy",'rb'))
            # self.scale_fn = lambda x: jnp.concatenate(
            #     [(x[:, :, :self.scaler.mean_.shape[0]] - self.scaler.mean_) / self.scaler.scale_,
            #       x[:, :, self.scaler.mean_.shape[0]:]], axis=2
            # )
            self.scale_fn = lambda x: (x - self.scaler.mean_) / self.scaler.scale_
        else:
            self.scaler = None
            self.scale_fn = lambda x: x

        if self.strategy_prediction in ("fit", "regression"):
            self.apply_strategy_prediction_fn = Predictor(
                hidden_channels=    self.hidden_channels,
                heads=              self.heads,
                layers=             self.layers,
                strategy_sampling=  "compute",
                method=             self.strategy_prediction
            )
        elif self.strategy_prediction is not None:
            self.apply_strategy_prediction_fn = eval("self.apply_prediction_" + self.strategy_prediction)
        else:
            self.apply_strategy_prediction_fn = lambda *args: (None, None, None, None, None, None)
        
        if self.augment:
            self.augment_fn = self.add_reference
        else:
            self.augment_fn = lambda *args: args[2]
        
        # Output MLPs

        self.mlp_graph = nn.Sequential([
            nn.Dense(2 * self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels // 2),
            nn.relu,
            nn.Dense(3)
        ])

        self.mlp_nodes = nn.Sequential([
            nn.Dense(2 * self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels // 2),
            nn.relu,
            nn.Dense(4)
        ])

        self.mlp_edges = nn.Sequential([
            nn.Dense(2 * self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels),
            nn.relu,
            nn.Dense(self.hidden_channels // 2),
            nn.relu,
            nn.Dense(2)            
        ])

        self.softmax = lambda x: nn.activation.softmax(x, axis=1)
    
    def apply_prediction_none(self, x, mask, *args):
            rng = jax.random.PRNGKey(0)
            rng, init_rng = jax.random.split(rng)
            r1 = -5
            points = jax.random.uniform(init_rng, shape=[x.shape[0], 3], minval=r1, maxval=-r1)
            log_errors = jnp.zeros([x.shape[0], 3])
            return None, None, None, points, log_errors, None

    def apply_prediction_perfect(self, x, mask, true_jet, true_trk, *args):
            assert(true_jet is not None and true_trk is not None)
            points = true_jet
            log_errors = jnp.zeros([x.shape[0], 3]) 
            return None, None, None, points, log_errors, None

    def add_reference(self, x, new_ref, g, mask):
        batch_size, n_tracks, _ = x.shape

        x_prime = self.extrapolator(x, jax.lax.stop_gradient(new_ref))
        if self.points_as_features:
            x_points = jnp.repeat(new_ref, n_tracks, axis=0).reshape(batch_size, n_tracks, 3)
            x_prime = jnp.concatenate([x_prime, x_points], axis=2)
            
        x_prime = self.scale_fn(x_prime)

        t_prime, g_prime = self.preprocessor(x_prime, mask)
        repr_track = jnp.concatenate([g, g_prime], axis=2)

        return repr_track

    def __call__(self, x, mask, true_jet, true_trk, n_tracks, jet_phi, jet_theta):
        # mask = mask | ~mask  # FIXME care
        assert(x.ndim == 3)  # n_jets, n_tracks, n_features
        batch_size, n_tracks, _ = x.shape

        x_ = x * mask

        if self.points_as_features:
            x_ = jnp.concatenate([x_, jnp.zeros(shape=(batch_size, n_tracks, 3))], axis=2)

        x_scaled = self.scale_fn(x_)
        if self.one_hot:
            ids = jnp.argsort(x_scaled[:, :, 0], axis=1)[:, ::-1]
            ids = jax.nn.one_hot(ids, n_tracks)
            # print(ids.shape)
            # exit(0)

            # ids = jnp.identity(n_tracks).reshape(1, n_tracks, n_tracks)
            # ids = ids.repeat(batch_size, axis=0)
            x_scaled = jnp.concatenate([x_scaled, ids], axis=2)


        t, g = self.preprocessor(x_scaled, mask)
            
        out_preds = self.apply_strategy_prediction_fn(x, mask, true_jet, true_trk, n_tracks, jet_phi, jet_theta)
        _, _, _, out_mean, out_var, out_chi = out_preds

        repr_track = self.augment_fn(x, out_mean, g, mask)

        # Compute jet-level representation
        repr_jet, _ = self.pool(repr_track, mask)

        # Compute track-level representation
        repr_jet_exp = jnp.repeat(repr_jet, n_tracks, axis=0).reshape(batch_size, n_tracks, -1)
        t = repr_track
        repr_track = jnp.concatenate([repr_track, repr_jet_exp], axis=2)
        
        # Compute edge-lavel representation
        a1 = jnp.repeat(t, n_tracks,axis=1)
        a2 = jnp.repeat(t, n_tracks,axis=0).reshape(batch_size , n_tracks**2, -1)
        a3 = jnp.repeat(repr_jet,n_tracks**2,axis=0).reshape(batch_size,n_tracks**2,-1)
        repr_vtx  = jnp.concatenate((a1,a2,a3), axis=2)

        # Obtain output probabilities 
        out_graph = self.mlp_graph(repr_jet)
        assert(out_graph.shape == (batch_size, 3))
        out_graph = nn.softmax(out_graph, axis=1)

        out_nodes = self.mlp_nodes(repr_track)
        assert(out_nodes.shape == (batch_size, n_tracks, 4))
        out_nodes = nn.softmax(out_nodes, axis=2)
        
        out_edges = self.mlp_edges(repr_vtx)
        assert(out_edges.shape == (batch_size, n_tracks**2, 2))
        out_edges = nn.softmax(out_edges, axis=2)

        return out_graph, out_nodes, out_edges, out_mean, out_var, out_chi

    def loss(self, out, batch, mask_nodes, mask_edges):
        out_graph, out_nodes, out_edges, out_mean, out_var, out_chi = out

        out_nodes = jnp.where(mask_nodes, out_nodes, 0)
        mask_edges = mask_edges.reshape(-1, 225, 1)
        out_edges = jnp.where(mask_edges, out_edges, 0)

        loss_graph = jnp.reshape(categorical_cross_entropy(jnp.reshape(batch['jet_y'], (-1,1,3)), jnp.reshape(out_graph, (-1,1,3)), None), (-1, 1))
        loss_graph = jnp.mean(loss_graph)
        
        loss_nodes = jnp.reshape(categorical_cross_entropy(batch['trk_y'], out_nodes, None), (-1, 15, 1))
        loss_nodes = jnp.mean(loss_nodes, where=mask_nodes)
        
        loss_edges = jnp.reshape(categorical_cross_entropy(batch['edge_y'], out_edges, None), (-1, 225, 1))
        loss_edges = jnp.mean(loss_edges, where=mask_edges)

        if out_mean is not None: 
            loss_predictor = jnp.sqrt(jnp.sum((batch['jet_vtx']-out_mean)**2, axis=1)).reshape(-1,1)
            loss_predictor = jnp.mean(loss_predictor)
        else:
            loss_predictor = 0

        loss = loss_graph + 0.5 * loss_nodes + 1.5 * loss_edges + 0.1 * loss_predictor

        losses_aux = (loss_graph, loss_nodes, loss_edges, loss_predictor)

        return loss, losses_aux
    # print(state.params['preprocessor']['enc_layer_0']['lin1']['kernel'].shape)
    # print(state.params['preprocessor']['enc_layer_0']['lin1']['bias'].shape)




