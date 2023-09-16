import sys
sys.path.append("../utils/")

from flax import linen as nn  
import jax.numpy as jnp
import jax

from utils.layers import Encoder 
from models.IDEncoder import IDEncoder

class PreProcessor(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    architecture: str
    use_encodings: bool
    encoding_strategy: str
    num_graphs: int 

    def setup(self):
        last_dim = self.hidden_channels - 15 * self.use_encodings * (self.num_graphs == 1)

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

        self.rpgnn = IDEncoder(pooling_strategy=self.encoding_strategy)

    # FIXME remove last arg, shouldn't be needed with the IDEncoder file
    def __call__(self, x, mask=None, embed=None):
        if mask is None:
            mask = jnp.ones((x.shape[0], 1))

        x = x * mask
        if embed is None:
            t = self.track_init(x)
        else: 
            t = embed
        t = t * mask

        if self.use_encodings:
            g = self.rpgnn(self.encoder, x, t, mask)
        else:
            g = self.encoder(t, mask=mask)
        
        g = g * mask

        return t, g

