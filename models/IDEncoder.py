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
            self.pooling_fn = lambda encoder, x, t, mask: encoder(t, mask=mask)
        else:
            self.pooling_fn = eval("self.encode_" + self.pooling_strategy)
        
        if self.pooling_strategy == "simple_random":
            #  if not hasattr(self, "seed"):
            #     self.seed = np.random.randint(0, high=42)
            self.key = jax.random.PRNGKey(self.seed)
            self.key2 = jax.random.PRNGKey(self.seed+2)
            print("seed =", self.seed)

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
    
    def encode_simple_eye(self, encoder, x, t, mask):
        ids = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])][::-1] * x.shape[0], axis=0)

        t_rp = self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)

    def encode_simple_pt(self, encoder, x, t, mask):
        ids = self.get_encodings_feature(x, t, mask, decreasing=True)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)

        t_rp = ids * mask # self.get_encodings(t, ids) * mask
        t_rp_inv = self.get_encodings(t, ids_rev) * mask
            
        g_inv = encoder(t_rp_inv, mask=mask)
        g = encoder(t_rp, mask=mask)

        return 0.5 * (g + g_inv)
    
    def encode_pt(self, encoder, x, t, mask):
        ids = self.get_encodings_feature(x, t, mask, decreasing=True)
        ids_rev = self.get_encodings_feature(x, t, mask, decreasing=False)

        t_rp = ids * mask # self.get_encodings(t, ids) * mask
        t_rp_inv = ids_rev * mask #self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv)  
    
    def encode_d0(self, encoder, x, t, mask):
        ids = self.get_encodings_feature(x, t, mask, decreasing=True, var_id=1)
        ids_rev = self.get_encodings_feature(x, t, mask, decreasing=False, var_id=1)

        t_rp = ids * mask # self.get_encodings(t, ids) * mask
        t_rp_inv = ids_rev * mask #self.get_encodings(t, ids_rev) * mask
            
        g = encoder(t_rp, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return 0.5 * (g + g_inv) 
    
    def encode_simple_random(self, encoder, x, t, mask):
        identities = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        # key_t = 0 #time.time_ns() % 100
        ids = jax.random.permutation(self.key, identities, axis=1)
        ids2 = jax.random.permutation(self.key2, identities, axis=1)

        t_rp = self.get_encodings(t, identities) * mask
        t_rp2 = self.get_encodings(t, ids2) * mask
        t_rp_inv = self.get_encodings(t, ids) * mask

        g = encoder(t_rp, mask=mask)
        g_2 = encoder(t_rp2, mask=mask)
        g_inv = encoder(t_rp_inv, mask=mask)

        return (g + g_2 + g_inv) / 3
    
    def encode_best(self, encoder, x, t, mask):
        ids_eye = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])][::-1] * x.shape[0], axis=0)
        ids_pt = self.get_encodings_feature(x, t, mask, decreasing=True)
        ids_rho = self.get_encodings_feature(x, t, mask, decreasing=True, var_id=5)
        ids_deltar = self.get_encodings_feature(x, t, mask, decreasing=True, var_id=7)
        
        t_eye = self.get_encodings(t, ids_eye) * mask
        t_rev = self.get_encodings(t, ids_rev) * mask
        t_pt = ids_pt * mask
        t_rho = ids_rho * mask
        t_deltar = ids_deltar * mask
        
        g_eye = encoder(t_eye, mask=mask)
        g_rev = encoder(t_rev, mask=mask)
        g_pt = encoder(t_pt, mask=mask)
        g_rho = encoder(t_rho, mask=mask)
        g_deltar = encoder(t_deltar, mask=mask)
        return 0.2 * (g_eye + g_rev + g_pt + g_rho + g_deltar)
        
    def encode_eye_pt(self, encoder, x, t, mask):
        ids_eye = jnp.stack([jnp.arange(0, x.shape[1])] * x.shape[0], axis=0)
        ids_rev = jnp.stack([jnp.arange(0, x.shape[1])][::-1] * x.shape[0], axis=0)
        ids_pt = self.get_encodings_feature(x, t, mask, decreasing=True)
        
        t_eye = self.get_encodings(t, ids_eye) * mask
        t_rev = self.get_encodings(t, ids_rev) * mask
        t_pt = ids_pt * mask
      
        g_eye = encoder(t_eye, mask=mask)
        g_rev = encoder(t_rev, mask=mask)
        g_pt = encoder(t_pt, mask=mask)
      
      
        return (g_eye + g_rev + g_pt) / 3

    def encode_augmented(self, encoder, x, t, mask):
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

    def __call__(self, encoder, x, t, mask=None):
        return self.pooling_fn(encoder, x, t, mask)


