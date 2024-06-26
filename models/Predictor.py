import sys
sys.path.append("../utils/")

import jax.numpy as jnp

from flax import linen as nn  
import jax
import datetime
from models.PreProcessor import PreProcessor
from models.Regression import Regression
from utils.fit import ndive
import utils.data_format as daf
from utils.losses import *


class Predictor(nn.Module):
    hidden_channels: int
    layers: int
    heads: int
    strategy_weights: str
    strategy_sampling: str
    use_ghost_track: bool
    activation: str
    method: str
    encoding_strategy: str
    seed: int

    def setup(self):
        self.preprocessor = PreProcessor(
            hidden_channels = self.hidden_channels,
            layers = self.layers,
            heads = self.heads,
            architecture="post",
            encoding_strategy=self.encoding_strategy,
            num_graphs=1,
            seed = self.seed
        )

        if self.strategy_weights == "compute":
            self.nn_sample = nn.Sequential([
                nn.Dense(features=self.hidden_channels, param_dtype=jnp.float64),
                nn.relu,
                nn.Dense(features=1, param_dtype=jnp.float64)
            ])
        
        if self.activation == "sigmoid":
            self.activation_fn = nn.sigmoid
        elif self.activation == "softmax":
            self.activation_fn = lambda x: nn.activation.softmax(x, axis=1)
        elif self.activation is None:
            self.activation_fn = lambda x: x

        self.apply_strategy_weights_fn = eval("self.apply_strategy_weights_" + self.strategy_weights)
        if self.strategy_sampling is not None:
            self.sample_fn = eval("self.sample_" + self.strategy_sampling)
            self.sample_fn = lambda *args: jax.lax.stop_gradient(self.sample_fn(*args))
        else:
            self.sample_fn = lambda *args: args

        if self.method == "regression":
            self.fitting_method = Regression(hidden_channels=self.hidden_channels)
        else:
            self.fitting_method = lambda x, _, wv, mask: ndive(x, jnp.where(mask.squeeze(), wv.squeeze(), 1e-100), jnp.zeros([x.shape[0], 3]))

    def apply_strategy_weights_none(self, repr_track, mask, true_jet, true_trk):
        weights = jnp.ones(mask.shape)
        weights = (weights * mask)
        return weights

    def apply_strategy_weights_perfect(self, repr_track, mask, true_jet, true_trk):
        true_trk = true_trk.reshape(-1 ,3)
        true_jet = true_jet.repeat(repr_track.shape[1], axis=0)
        # weights = (true_jet[:, 2] == true_trk[:, 2])
        weights = (true_jet[:, 0] == true_trk[:, 0]) & (true_jet[:, 1] == true_trk[:, 1]) & (true_jet[:, 2] == true_trk[:, 2])
        weights = weights.reshape(*mask.shape)

        weights = (mask & weights).astype(jnp.float64)

        return weights

    def apply_strategy_weights_compute(self, repr_track, mask, true_jet, true_trk):

        weights = self.nn_sample(repr_track)
        weights = jnp.where(mask, weights, jnp.array([-jnp.inf]))
        weights = self.activation_fn(weights)
        
        return weights

    def sample_bernoulli(self, w):
        current_date = datetime.datetime.now()
        key = jax.random.PRNGKey(current_date.second)
        selection = jax.random.bernoulli(key, w) # jnp.sqrt(w))
        w = selection * w
        w = nn.activation.softmax(w, axis=1)
        return w

    def add_ghost_track(self, x, mask, n_tracks, jet_phi, jet_theta):
        num_jets, max_num_tracks, _ = x.shape

        new_track = jnp.stack([
            jnp.array([
                0.,
                0.,
                0.,
                jet_phi,
                jet_theta,
                0.,
                0.,
                0.,
                0.,
                jnp.mean(d_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(z_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(p_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(t_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(r_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                0.,
                0.,
                jnp.log(pt),
                eta,
            ]) for n,d_o,z_o,p_o,t_o,r_o,pt,eta in zip(
                n_tracks,
                x[:,:,daf.JetData.TRACK_D0_ERR],
                x[:,:,daf.JetData.TRACK_Z0_ERR],
                x[:,:,daf.JetData.TRACK_PHI_ERR],
                x[:,:,daf.JetData.TRACK_THETA_ERR],
                x[:,:,daf.JetData.TRACK_RHO_ERR],
                x[:,0,daf.JetData.TRACK_JET_PT],
                x[:,0,daf.JetData.TRACK_JET_ETA],
            )
        ]).reshape(num_jets,1,daf.NUM_JET_INPUT_PARAMETERS)

        mask = jnp.concatenate([jnp.ones((x.shape[0], 1, 1)).astype(bool), mask], axis=1)
        x = jnp.concatenate((new_track, x[:,:,0:daf.NUM_JET_INPUT_PARAMETERS]),axis=1)
        
        return x, mask

    def __call__(self, x, mask, true_jet, true_trk, n_tracks, jet_phi, jet_theta, *args, **kwargs):
        num_jets, max_num_tracks, _ = x.shape

        if self.use_ghost_track:
            x, mask = self.add_ghost_track(x, mask, n_tracks, jet_phi, jet_theta)
        
        t, g = self.preprocessor(x, mask)

	        	
        if isinstance(self.fitting_method, Regression):	
            repr_track = jnp.concatenate([t, g], axis=2)	
        else:	
            repr_track = g	


        weights = self.apply_strategy_weights_fn(repr_track, mask, true_jet, true_trk)   
        weights = jnp.where(mask, weights, 1e-100)   
        out_mean, out_var, out_chi = self.fitting_method(x, repr_track, weights, mask)

        out_mean = jnp.clip(out_mean, a_min=-4000., a_max=4000.)
        out_mean = jnp.nan_to_num(out_mean, nan=4000., posinf=4000., neginf=-4000.)
        out_var = jnp.nan_to_num(out_var, nan=1000., posinf=1000., neginf=1000.)
        if not isinstance(self.fitting_method, Regression):
            out_var = jax.lax.stop_gradient(out_var)

        if out_chi is not None:
            out_chi = jnp.nan_to_num(out_chi, nan=1000., posinf=1000., neginf=1000.)
            out_chi = jax.lax.stop_gradient(out_chi)

        # weights = weights[:, return_idx]
        return None, None, None, out_mean, out_var, out_chi, weights

    def loss(self, out, batch, mask, mask_edges):
        _, _, _, out_mean, out_var, out_chi = out

        # loss_f = jnp.sqrt(jnp.sum((batch['jet_vtx']-out_mean)**2, axis=1))

        # valid = jnp.any(jnp.any(batch['trk_y'], axis=2), axis=1) # 250 x15 


        ##loss_f = jnp.mean(loss_f)#, where=valid) #, where=batch['jet_y'][:, 0] != 1)
        # loss_g = gaussian_neg_loglikelihood(batch['jet_vtx'], out_mean, jnp.exp(out_var))
        #  loss_f = squared_error(batch['jet_vtx'], out_mean)
        loss_f = jnp.abs(batch['jet_vtx'] - out_mean)
        loss_f = jnp.mean(loss_f)

        # loss_f = jnp.sqrt(jnp.sum((batch['jet_vtx']-out_mean)**2, axis=1))
        # weights = jnp.log(batch['x'][:, 0, 16])
        # loss_f = jnp.average(loss_f, weights=weights)

        loss = loss_f  #+ .2 * loss_g

        return loss, (0, 0, 0, loss_f)

