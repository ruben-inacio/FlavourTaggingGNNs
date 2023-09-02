import sys
sys.path.append("../utils/")

from flax import linen as nn  
import jax.numpy as jnp
try:    
    from FlavourTaggingGNNs.utils.layers import GlobalAttention
except ModuleNotFoundError:
    from utils.layers import GlobalAttention


class Regression(nn.Module):
    hidden_channels: int
    use_weights: bool

    def setup(self):
        self.gate_nn = nn.Sequential([	
                nn.Dense(features=self.hidden_channels, param_dtype=jnp.float64),	
                nn.relu,	
                nn.Dense(features=1, param_dtype=jnp.float64)	
        ])
        self.pool = GlobalAttention(gate_nn=self.gate_nn)
        
        self.mlp_mean = nn.Sequential([
            nn.Dense(self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(3, param_dtype=jnp.float64)
        ])
            
        self.mlp_var = nn.Sequential([
            nn.Dense(self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(self.hidden_channels, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(3, param_dtype=jnp.float64)
        ])

    def __call__(self, x, repr_track, weights, mask, *args):
        # TODO implement pre processor no?
        if self.use_weights:
            # pooled, _ = self.pool(weights * repr_track, mask=mask)
            pooled, _ = self.pool(weights * x, mask=mask)
        else:
            pooled, _ = self.pool(repr_track, mask=mask)
        out_mean = self.mlp_mean(pooled)
        out_var = self.mlp_var(pooled)
        
        return out_mean, out_var, None

