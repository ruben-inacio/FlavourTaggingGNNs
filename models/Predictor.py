import sys
sys.path.append("../utils/")

import jax.numpy as jnp

from flax import linen as nn  
from models.PreProcessor import PreProcessor
from models.Regression import Regression
from utils.fit import ndive
from utils.losses import squared_error
import jax
import datetime


class Predictor(nn.Module):
    hidden_channels: int
    layers: int
    heads: int
    strategy_sampling: str
    method: str

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
        true_jet = true_jet.repeat(repr_track.shape[1], axis=0)
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
        out_mean = jnp.nan_to_num(out_mean, nan=4000., posinf=4000., neginf=-4000.)
        out_var = jnp.nan_to_num(out_var, nan=1000., posinf=1000., neginf=1000.)

        if out_chi is not None:
            out_var = jnp.nan_to_num(out_var, nan=1000., posinf=1000., neginf=1000.)

        return None, None, None, out_mean, out_var, out_chi

    def loss(self, out, batch, mask, mask_edges):
        _, _, _, out_mean, out_var, out_chi = out

        loss_f = jnp.sqrt(jnp.sum((batch['jet_vtx']-out_mean)**2, axis=1))
        loss_f = jnp.mean(loss_f) #, where=batch['jet_y'][:, 0] != 1)
        
        # loss_mse = squared_error(batch['jet_vtx'], out_mean)

        # loss_f = jnp.sqrt(jnp.sum((batch['jet_vtx']-out_mean)**2, axis=1))
        # weights = jnp.log(batch['x'][:, 0, 16])
        # loss_f = jnp.average(loss_f, weights=weights)

        loss = loss_f # + .2 * loss_mse

        return loss, (0, 0, 0, loss_f)

