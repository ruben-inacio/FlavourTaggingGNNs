import pickle
from flax import linen as nn  

import jax.numpy as jnp
import jax
from utils.layers import GlobalAttention
from utils.losses import *
from models.PreProcessor import PreProcessor
from models.Predictor import Predictor
from utils.extrapolation import extrapolation


class TN1(nn.Module):
    hidden_channels :    int
    layers:              int
    heads:               int
    augment:             bool
    strategy_prediction: str
    points_as_features:  bool
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

            self.scale_fn = lambda x: (x - self.scaler.mean_) / self.scaler.scale_
        else:
            self.scaler = None
            self.scale_fn = lambda x: x

        if self.strategy_prediction in ("fit", "regression"):
            self.apply_strategy_prediction_fn = Predictor(
                hidden_channels=    self.hidden_channels,
                layers=             self.layers,
                heads=              self.heads,
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
        assert(x.ndim == 3)  # n_jets, n_tracks, n_features
        batch_size, n_tracks, _ = x.shape

        x_ = x * mask

        if self.points_as_features:
            x_ = jnp.concatenate([x_, jnp.zeros(shape=(batch_size, n_tracks, 3))], axis=2)

        x_scaled = self.scale_fn(x_)
        if self.one_hot:
            ids = jnp.argsort(x_scaled[:, :, 0], axis=1)[:, ::-1]
            ids = jax.nn.one_hot(ids, n_tracks)
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

