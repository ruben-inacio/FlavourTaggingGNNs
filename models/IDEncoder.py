import sys
sys.path.append("../utils/")

from flax import linen as nn  
import jax.numpy as jnp
import jax
import time

class IDEncoder(nn.Module):
    pooling_strategy: str

    def setup(self):
        self.pooling_fn = eval("self.encode_" + self.pooling_strategy)
        
    def get_encodings(self, x, ids):
        ids = jax.nn.one_hot(ids, ids.shape[1])
        x = jnp.concatenate([x, ids], axis=2)
        return x

    def get_ids(self, x, mask, reverse_mode=True, var_id=0):
        x = x[:, :, var_id]
        x = jnp.where(mask[:, :, var_id], x, 0)
        ids = jnp.argsort(x, axis=1)
        if reverse_mode:
            ids = ids[:, ::-1]

        return ids

    def encode_simple_reversed(self, encoder, x, t, mask):
        ids = self.get_ids(x, mask, reverse_mode=True)
        ids_rev = self.get_ids(x, mask, reverse_mode=False)

        t_rp = self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)

    def encode_simple_eye(self, encoder, x, t, mask):
        ids = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])][::-1] * x.shape[0], axis=0)

        t_rp = self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)

    def encode_simple_pt(self, encoder, x, t, mask):
        ids = self.get_ids(x, mask, reverse_mode=True)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)

        t_rp = self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)
    
    def encode_simple_random(self, encoder, x, t, mask):
        identities = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        key_t = time.time_ns() % 100
        key = jax.random.PRNGKey(key_t)
        ids = jax.random.permutation(key, identities, axis=1)
        ids_rev = jax.random.permutation(key, ids, axis=1)

        t_rp = self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask

        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)
    
    def __call__(self, encoder, x, t, mask=None):
        return self.pooling_fn(encoder, x, t, mask)


