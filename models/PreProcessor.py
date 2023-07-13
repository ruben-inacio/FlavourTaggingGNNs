from flax import linen as nn  
from utils.layers import Encoder


class PreProcessor(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    architecture: str

    def setup(self):
        self.track_init = nn.Sequential([
            nn.Dense(features=self.hidden_channels),
            nn.relu,
            nn.Dense(features=self.hidden_channels),
            nn.relu,
            nn.Dense(features=self.hidden_channels),
            nn.relu,
            nn.Dense(features=self.hidden_channels),
        ])
        
        self.encoder = Encoder(
            hidden_channels=self.hidden_channels, 
            heads=self.heads, 
            layers=self.layers, 
            architecture=self.architecture
        )
        
    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones((x.shape[0], 1))

        x = x * mask

        t = self.track_init(x)
        t = t * mask
        # TODO bnorm

        g = self.encoder(t, mask=mask)
        g = g * mask

        return t, g

