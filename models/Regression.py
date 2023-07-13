from flax import linen as nn  
from utils.layers import GlobalAttention


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

