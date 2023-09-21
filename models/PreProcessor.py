import sys
sys.path.append("../utils/")

from flax import linen as nn  
import jax.numpy as jnp
import jax
import datetime

from utils.layers import Encoder 
from models.IDEncoder import IDEncoder


class PreProcessor(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    architecture: str
    encoding_strategy: str
    num_graphs: int 
    seed: int 

    def setup(self):
        last_dim = self.hidden_channels - 15 * (self.encoding_strategy != None) * (self.num_graphs == 1)

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
            hidden_channels=self.hidden_channels + (15 * self.num_graphs if self.num_graphs != 1 else 0), 
            heads=self.heads, 
            layers=self.layers, 
            architecture=self.architecture
        )


        self.rpgnn = IDEncoder(pooling_strategy=self.encoding_strategy, seed=self.seed)


    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones((x.shape[0], 1))

        x = x * mask
        # TODO (try same mlp for both graphs in augmentation)
        # maybe use num graphs and fori_loop ?

            
        t = self.track_init(x)
        t = t * mask

        g = self.rpgnn(self.encoder, x, t, mask)
        g = g * mask

        return t, g

