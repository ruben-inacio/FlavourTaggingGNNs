import pickle
from flax import linen as nn  

import jax.numpy as jnp
import jax
import datetime

try:
    from FlavourTaggingGNNs.utils.layers import GlobalAttention, EncoderLayer, Encoder
    from FlavourTaggingGNNs.utils.losses import *
    from FlavourTaggingGNNs.models.PreProcessor import PreProcessor
    from FlavourTaggingGNNs.models.Predictor import Predictor
    from FlavourTaggingGNNs.utils.extrapolation import extrapolation
except ModuleNotFoundError:
    from utils.layers import GlobalAttention, EncoderLayer, Encoder
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
    errors_as_features:  bool
    scale:               bool
    strategy_encodings:  str

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
        # self.preprocessor2 = PreProcessor(
        #     hidden_channels = self.hidden_channels,
        #     layers = self.layers,
        #     heads = self.heads,
        #     architecture="post"
        # )
        
        self.gate_nn = nn.Dense(features=1, param_dtype=jnp.float64)
        self.pool = GlobalAttention(gate_nn=self.gate_nn)

        self.extraplator = None
        if self.augment:
            self.extrapolator = extrapolation
            # self.augm_encoder = Encoder(
            #     hidden_channels=self.hidden_channels,
            #     heads=self.heads,
            #     layers=self.layers,
            #     architecture="post"
            # )
            # self.augm_lin = nn.Dense(features=self.hidden_channels)	
            self.augment_fn = self.add_reference	
        else:	
            self.augment_fn = lambda *args: args[4]

        if self.scale:  # TODO FIXME out of date
            if self.points_as_features:
                self.scaler=pickle.load(open("../training_data/scaler_30jun.npy",'rb'))
            else:
                self.scaler=pickle.load(open("../training_data/scaler_28jun.npy",'rb'))

            self.scale_fn = lambda x: (x - self.scaler.mean_) / self.scaler.scale_
        else:
            self.scaler = None
            self.scale_fn = lambda x: x

        if self.strategy_prediction in ("fit", "regression"):
            # FIXME 
            # self.apply_strategy_prediction_fn = Predictor(
            #     hidden_channels=    self.hidden_channels,
            #     layers=             self.layers,
            #     heads=              self.heads,
            #     strategy_sampling=  "compute",
            #     method=             self.strategy_prediction
            # )
            self.apply_strategy_prediction_fn = Predictor(
                hidden_channels=         self.hidden_channels, #64,
                layers=                  self.layers,  #3,
                heads=                   self.heads,  #2,
                strategy_sampling=       "none",
                strategy_weights=        "compute",
                use_ghost_track=         False,
                activation = "softmax",
                method=                  self.strategy_prediction
            )
        elif self.strategy_prediction is not None:
            self.apply_strategy_prediction_fn = eval("self.apply_prediction_" + self.strategy_prediction)
        else:
            self.apply_strategy_prediction_fn = lambda *args: (None, None, None, None, None, None)
        
        if self.strategy_encodings is not None:
            self.encodings_fn = eval("self.encodings_" + self.strategy_encodings)
        # Output MLPs

        self.mlp_graph = nn.Sequential([
            nn.Dense(2 * self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels // 2, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(3, param_dtype=jnp.float64)
        ])

        self.mlp_nodes = nn.Sequential([
            nn.Dense(2 * self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels // 2, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(4, param_dtype=jnp.float64)
        ])

        self.mlp_edges = nn.Sequential([
            nn.Dense(2 * self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels // 2, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(2, param_dtype=jnp.float64)            
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

    def encodings_positional(self, x, ids):
    def get_pe_batch(x, ids):
        _, max_len, d_model = x.shape
        pe = np.zeros(x.shape)
        position = ids[:, :, None]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))[:max_len]
        div_term = np.repeat(div_term[:, None], 10, axis=1).T[:, :, None]
        # print(ids.shape, position.shape, div_term.shape)
        pe[:, :, 0::2] = np.sin(position * div_term)
        pe[:, :, 1::2] = np.cos(position * div_term)
        x = x + pe
        return x
        
    def encodings_one_hot(self, x, ids):
        ids = jax.nn.one_hot(ids, ids.shape[1])
        x = jnp.concatenate([x, ids], axis=2)
        return x

    def get_ids(self, x, mask, reverse_mode=True):
        x = x[:, :, 0]
        x = jnp.where(mask[:, :, 0], x, 0)
        ids = jnp.argsort(x, axis=1)
        if reverse_mode:
            ids = ids[:, ::-1]
        return ids
    
    def add_reference(self, x, new_ref, new_ref_errors, t, g, mask, thresholds):
        batch_size, n_tracks, _ = x.shape
        x_prime = self.extrapolator(x, jax.lax.stop_gradient(new_ref))
        x_prime = x_prime * mask
        if self.points_as_features:
            if new_ref_errors.ndim == 3:
                new_ref_errors = jax.lax.map(jnp.diag, new_ref_errors)
            x_points = jnp.repeat(new_ref, n_tracks, axis=0).reshape(batch_size, n_tracks, 3)
            x_errors = jnp.repeat(new_ref_errors, n_tracks, axis=0).reshape(batch_size, n_tracks, 3)
            x_prime = jnp.concatenate([x_prime, x_points], axis=2)
            if self.errors_as_features:
                x_prime = jnp.concatenate([x_prime, x_errors], axis=2)
                
        x_prime = self.scale_fn(x_prime)
        
        if self.strategy_encodings is not None:
            ids = self.get_ids(x_prime, mask, reverse_mode=False)
        
        if self.strategy_encodings == "one_hot":
            x_prime = self.encodings_fn(x_prime, ids)

        t_prime, g_prime = self.preprocessor(x_prime, mask)
        
        if self.strategy_encodings == "positional":
            g_prime = self.encodings_fn(g_prime, ids)

        # t_mixed = jnp.concatenate([t, t_prime], axis=1)	
        # mask_mixed = jnp.concatenate([mask, mask], axis=1)	
        # g_mixed = self.augm_encoder(t_mixed, mask=mask_mixed)	
        # g_mixed = jnp.concatenate([g_mixed[:, :n_tracks, :], g_mixed[:, n_tracks:, :]], axis=2)	
        # g_mixed = self.augm_lin(g_mixed)	
        # repr_track = jnp.concatenate([g, g_prime, g_mixed], axis=2)
        repr_track = jnp.concatenate([g, g_prime], axis=2)

        return repr_track

    def __call__(self, x, mask, true_jet, true_trk, n_tracks, jet_phi, jet_theta, fix=False):
        assert(x.ndim == 3)  # n_jets, n_tracks, n_features
        batch_size, max_tracks, _ = x.shape

        x_ = x * mask

        if self.points_as_features:
            x_ = jnp.concatenate([x_, jnp.zeros(shape=(batch_size, max_tracks, 3))], axis=2)
            if self.errors_as_features:
                x_ = jnp.concatenate([x_, jnp.zeros(shape=(batch_size, max_tracks, 3))], axis=2)


        x_scaled = self.scale_fn(x_)
        
        thresholds=None
        if self.strategy_encodings is not None:
            ids = self.get_ids(x_scaled, mask, reverse_mode=True)
        
        if self.strategy_encodings == "one_hot":
            x_scaled = self.encodings_fn(x_scaled, ids)

        t, g = self.preprocessor(x_scaled, mask)

        if self.strategy_encodings == "positional":
            g = self.encodings_fn(g, ids)
            
        if not fix:
            out_preds = self.apply_strategy_prediction_fn(x, mask, true_jet, true_trk, n_tracks, jet_phi, jet_theta)
        else:
            out_preds = jax.lax.stop_gradient(self.apply_strategy_prediction_fn(x, mask, true_jet, true_trk, n_tracks, jet_phi, jet_theta))
        _, _, _, out_mean, out_var, out_chi = out_preds
        
        repr_track = self.augment_fn(x, out_mean, out_var, t, g, mask, thresholds)
        # 
        # key = jax.random.PRNGKey(datetime.datetime.now().second)
        # out_var_diag = jax.lax.map(jnp.diag, out_var)
        # out_samples = jax.lax.stop_gradient(jax.random.uniform(key, shape=out_mean.shape, minval=out_mean-out_var_diag, maxval=out_mean+out_var_diag))
        # repr_track = self.augment_fn(x, out_samples, out_var, t, g, mask, thresholds=thresholds)
        # 

        # Compute jet-level representation
        repr_jet, _ = self.pool(repr_track, mask)

        # Compute track-level representation
        repr_jet_exp = jnp.repeat(repr_jet, max_tracks, axis=0).reshape(batch_size, max_tracks, -1)
        t = repr_track
        repr_track = jnp.concatenate([repr_track, repr_jet_exp], axis=2)
        
        # Compute edge-lavel representation
        a1 = jnp.repeat(t, max_tracks,axis=1)
        a2 = jnp.repeat(t, max_tracks,axis=0).reshape(batch_size , max_tracks**2, -1)
        a3 = jnp.repeat(repr_jet,max_tracks**2,axis=0).reshape(batch_size,max_tracks**2,-1)
        repr_vtx  = jnp.concatenate((a1,a2,a3), axis=2)

        # Obtain output probabilities 
        out_graph = self.mlp_graph(repr_jet)
        assert(out_graph.shape == (batch_size, 3))
        out_graph = nn.softmax(out_graph, axis=1)

        out_nodes = self.mlp_nodes(repr_track)
        assert(out_nodes.shape == (batch_size, max_tracks, 4))
        out_nodes = nn.softmax(out_nodes, axis=2)
        
        out_edges = self.mlp_edges(repr_vtx)
        assert(out_edges.shape == (batch_size, max_tracks**2, 2))
        out_edges = nn.softmax(out_edges, axis=2)
        # FIXME may work only on analysis
        return out_graph, out_nodes, out_edges, out_mean, out_var, out_chi, #t

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

