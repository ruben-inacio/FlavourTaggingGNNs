import sys
sys.path.append("../utils/")

from flax import linen as nn  
import jax.numpy as jnp
import jax

from utils.layers import Encoder 

class PreProcessor(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    architecture: str
    use_encodings: bool
    num_graphs: int 

    def setup(self):
        last_dim = self.hidden_channels - 15 * self.use_encodings * self.num_graphs

        self.track_init = nn.Sequential([
            nn.Dense(features=self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=last_dim, param_dtype=jnp.float64),
            nn.sigmoid
        ])
        # self.norm = nn.RMSNorm()

        self.encoder = Encoder(
            hidden_channels=self.hidden_channels, 
            heads=self.heads, 
            layers=self.layers, 
            architecture=self.architecture
        )

    def get_encodings(self, x, ids):
        ids = jax.nn.one_hot(ids, ids.shape[1])
        x = jnp.concatenate([x, ids], axis=2)
        return x

    def get_ids(self, x, mask, reverse_mode=True, var_id=0):
        x = x[:, :, 0]
        x = jnp.where(mask[:, :, var_id], x, 0)
        ids = jnp.argsort(x, axis=1)
        if reverse_mode:
            ids = ids[:, ::-1]
        # ids = jnp.arange(x.shape[1])
        return ids

    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones((x.shape[0], 1))

        x = x * mask

        t = self.track_init(x)
        t = t * mask

        if self.use_encodings:
            ids = self.get_ids(x, mask, reverse_mode=True)
            ids_rev = self.get_ids(x, mask, reverse_mode=False)

            t_rp = self.get_encodings(t, ids)
            t_rp_inv = self.get_encodings(t, ids_rev)
            
            g = self.encoder(t_rp, mask=mask)
            g_inv = self.encoder(t_rp_inv, mask=mask)

            g = 0.5 * (g + g_inv)

        else:
            g = self.encoder(t, mask=mask)
            g = g * mask

        return t, g

