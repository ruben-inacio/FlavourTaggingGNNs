import sys
sys.path.append("../utils/")

from flax import linen as nn  
import jax.numpy as jnp
import jax
import time
import datetime
import numpy as np


class IDEncoder(nn.Module):
    pooling_strategy: str
    seed: int = 42

    def setup(self):
        if self.pooling_strategy is None:
            self.pooling_fn = lambda encoder, x, t, mask, *args: encoder(t, mask=mask)
        else:
            self.pooling_fn = eval("self.encode_" + self.pooling_strategy)

    def get_encodings(self, x, ids):
        ids = jax.nn.one_hot(ids, ids.shape[1])
        x = jnp.concatenate([x, ids], axis=2)
        return x

    def get_encodings_feature(self, x, t, mask, decreasing=True, var_id=0):
        x = jax.lax.stop_gradient(x)
        x = x[:, :, var_id]

        #if decreasing:
        x = jnp.where(mask[:, :, 0], x, 0)
        #else:
        #    x = jnp.where(mask[:, :, 0], x, jnp.inf) 

        idx_argsort = jnp.argsort(x, axis=1)
        if decreasing:
            idx_argsort = idx_argsort[:, ::-1]
    
        idx_inv_argsort = jnp.argsort(idx_argsort, axis=1)
        eye = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        eye = jax.nn.one_hot(eye, eye.shape[1])

        sorted_t = t[:, idx_argsort][jnp.diag_indices(t.shape[0])]
        sorted_t = jnp.concatenate([sorted_t, eye], axis=2)
        res = sorted_t[:, idx_inv_argsort][jnp.diag_indices(sorted_t.shape[0])]

        return res
    
    def encode_eye(self, encoder, x, t, mask, *args, **kwargs):
        ids = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        t_rp = self.get_encodings(t, ids) * mask
            
        g = encoder(t_rp, mask=mask)

        return g

    def encode_random(self, encoder, x, t, mask, *args, **kwargs):
        ids = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        key = jax.random.PRNGKey(time.time_ns() % 100)
        ids = jax.random.permutation(key, ids)

        t_rp = self.get_encodings(t, ids) * mask
            
        g = encoder(t_rp, mask=mask)

        return g
    
    def encode_pt_simpl(self, encoder, x, t, mask, *args, **kwargs):
        ids = self.get_encodings_feature(x, t, mask, decreasing=True)

        t_rp = ids * mask # self.get_encodings(t, ids) * mask
            
        g = encoder(t_rp, mask=mask)

        return g
    # Old..
    def encode_simple_eye(self, encoder, x, t, mask, *args, **kwargs):
        ids = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])][::-1] * x.shape[0], axis=0)

        t_rp = self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)

    def encode_simple_pt(self, encoder, x, t, mask, *args, **kwargs):
        ids = self.get_encodings_feature(x, t, mask, decreasing=True)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)

        t_rp = ids * mask # self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask
            
        g_inv = encoder(t_rp_inv, mask=mask)
        g = encoder(t_rp, mask=mask)

        return 0.5 * (g + g_inv)
    
    def encode_simple_random(self, encoder, x, t, mask, *args, **kwargs):
        identities = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        key = jax.random.PRNGKey(self.seed)
        ids = jax.random.permutation(self.key, identities, axis=1)

        t_rp = self.get_encodings(t, identities) * mask
        t_rp_inv = self.get_encodings(t, ids) * mask

        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)
 
    def encode_random_dynamic(self, encoder, x, t, mask, offset, *args, **kwargs):
        identities = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        # key = jax.random.PRNGKey(time.time_ns() % 100)
        # key2 = jax.random.PRNGKey(datetime.datetime.now().second)
        key = jax.random.PRNGKey(self.seed-offset)
        key2 = jax.random.PRNGKey(self.seed+1+offset)
        ids = jax.random.permutation(key, identities, axis=1)
        ids2 = jax.random.permutation(key2, identities, axis=1)

        t_rp = self.get_encodings(t, ids2) * mask
        t_rp_inv = self.get_encodings(t, ids) * mask

        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)

    def encode_random_fixed(self, encoder, x, t, mask, *args, **kwargs):
        identities = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        key = jax.random.PRNGKey(self.seed)
        key2 = jax.random.PRNGKey(self.seed+1)
        ids = jax.random.permutation(key, identities, axis=1)
        ids2 = jax.random.permutation(key2, identities, axis=1)

        t_rp = self.get_encodings(t, ids2) * mask
        t_rp_inv = self.get_encodings(t, ids) * mask

        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)

    def encode_pt(self, encoder, x, t, mask, *args, **kwargs):
        ids = self.get_encodings_feature(x, t, mask, decreasing=True)
        ids_rev = self.get_encodings_feature(x, t, mask, decreasing=False)

        t_rp = ids * mask # self.get_encodings(t, ids) * mask
        t_rp_inv = ids_rev * mask #self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)  
    
    def encode_d0(self, encoder, x, t, mask, *args, **kwargs):
        ids = self.get_encodings_feature(x, t, mask, decreasing=True, var_id=1)
        ids_rev = self.get_encodings_feature(x, t, mask, decreasing=False, var_id=1)

        t_rp = ids * mask # self.get_encodings(t, ids) * mask
        t_rp_inv = ids_rev * mask #self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv) 
     
    def encode_augmented(self, encoder, x, t, mask, *args, **kwargs):
        _, n_tracks, _ = x.shape
        n_tracks = n_tracks // 2
        mask_ = mask[:, :n_tracks, :]

        ids = jnp.stack([jnp.arange(0, n_tracks)] * x.shape[0], axis=0)
        ids_rev = jnp.stack([jnp.arange(0, n_tracks)][::-1] * x.shape[0], axis=0)

        t_eye_1 = self.get_encodings(t[:, :n_tracks, :], ids) * mask_
        t_eye_2 = self.get_encodings(t[:, n_tracks:, :], ids) * mask_
    
        t_eye_rev_1 = self.get_encodings(t[:, :n_tracks, :], ids_rev) * mask_
        t_eye_rev_2 = self.get_encodings(t[:, n_tracks:, :], ids_rev) * mask_

        t_eye = jnp.concatenate([t_eye_1, t_eye_2], axis = 1)
        t_eye_rev = jnp.concatenate([t_eye_rev_1, t_eye_rev_2], axis = 1)

        g_eye = encoder(t_eye, mask=mask)
        g_rev = encoder(t_eye_rev, mask=mask)

        return 0.5 * (g_eye + g_rev)

    def __call__(self, encoder, x, t, mask=None, *args, **kwargs):
        return self.pooling_fn(encoder, x, t, mask, *args, **kwargs)


