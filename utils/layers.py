
from flax import linen as nn  
import jax.numpy as jnp


class GlobalAttention(nn.Module):
    gate_nn: nn.Module
    
    def setup(self):
        self.softmax = lambda x: nn.activation.softmax(x, axis=1)
    
    def __call__(self, x, mask=None):
        batch_size, length, _ = x.shape

        gate = self.gate_nn(x)

        if mask is not None:
            gate = jnp.where(mask, gate, jnp.array([-jnp.inf]))

        gate = self.softmax(gate)
        out = jnp.sum(x * gate, axis=1)

        return out, gate


class EncoderLayer(nn.Module):
    heads: int
    hidden_channels: int
    architecture:  str

    def setup(self):
        self.weights = None
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.heads, param_dtype=jnp.float64)
        self.norm1 = nn.LayerNorm(param_dtype=jnp.float64)
        self.lin1 = nn.Dense(features=4*self.hidden_channels, param_dtype=jnp.float64)

        self.norm2 = nn.LayerNorm(param_dtype=jnp.float64)
        self.lin2 = nn.Dense(features=self.hidden_channels, param_dtype=jnp.float64)
        self.fwd_fn = eval("self.fwd_" + self.architecture)
    
    def fwd_post(self, x, y):
        n = self.attn(x, y)
        x = x + n
        x = self.norm1(x)
        n = self.lin1(x)
        n = nn.relu(n)
        n = self.lin2(n)
        x = x + n
        x = self.norm2(x)
        return x

    def fwd_pre(self, x):
        n = self.norm1(x)
        n = self.attn(n)
        x = x + n
        n = self.norm2(x)
        n = self.lin1(n)
        n = nn.relu(n)
        n = self.lin2(n)
        x = x + n
        return x

    def fwd_postb2t(self, x):
        pass

    def __call__(self, x, y, mask=None):
        if mask is not None:
            x = x * mask
            y = y * mask
        x = self.fwd_fn(x, y)
        return x


class Encoder(nn.Module):
    hidden_channels: int
    heads: int
    layers: int
    architecture: str

    def setup(self):
        for i in range(self.layers):
            setattr(self, f"enc_layer_{i}", EncoderLayer(heads=self.heads, hidden_channels=self.hidden_channels, architecture=self.architecture))
   
        if self.architecture == "pre":
            self.norm3 = nn.LayerNorm()
        else:
            self.norm3 = lambda x: x

    def __call__(self, g, mask=None):
        for i in range(self.layers):
            g = getattr(self, f"enc_layer_{i}")(g, g, mask=mask)

        if mask is not None:
            g = g * mask
        
        g = self.norm3(g)
        if mask is not None:
            g = g * mask
        return g
 

class GATLayer(nn.Module):
    # Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial7/GNN_overview.html
    c_out : int  # Dimensionality of output features
    num_heads : int  # Number of heads, i.e. attention mechanisms to apply in parallel.
    concat_heads : bool = True  # If True, the output of the different heads is concatenated instead of averaged.
    alpha : float = 0.2  # Negative slope of the LeakyReLU activation.

    def setup(self):
        if self.concat_heads:
            assert self.c_out % self.num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out_per_head = self.c_out // self.num_heads
        else:
            c_out_per_head = self.c_out

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Dense(c_out_per_head * self.num_heads,
                                   kernel_init=nn.initializers.glorot_uniform())
        self.a = self.param('a',
                            nn.initializers.glorot_uniform(),
                            (self.num_heads, 2 * c_out_per_head))  # One per head


    def __call__(self, node_feats, adj_matrix, return_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            return_attn_probs - If True, the attention weights are returned after the forward pass
        """
        batch_size, num_nodes = node_feats.shape[0], node_feats.shape[1]

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.reshape((batch_size, num_nodes, self.num_heads, -1))

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # In order to take advantage of JAX's just-in-time compilation, we should not use
        # arrays with shapes that depend on e.g. the number of edges. Hence, we calculate
        # the logit for every possible combination of nodes. For efficiency, we can split
        # a[Wh_i||Wh_j] = a_:d/2 * Wh_i + a_d/2: * Wh_j.
        logit_parent = (node_feats * self.a[None,None,:,:self.a.shape[0]//2]).sum(axis=-1)
        logit_child = (node_feats * self.a[None,None,:,self.a.shape[0]//2:]).sum(axis=-1)
        attn_logits = logit_parent[:,:,None,:] + logit_child[:,None,:,:]
        attn_logits = nn.leaky_relu(attn_logits, self.alpha)

        # Mask out nodes that do not have an edge between them
        attn_logits = jnp.where(adj_matrix[...,None] == 1.,
                                attn_logits,
                                jnp.ones_like(attn_logits) * (-9e15))

        # Weighted average of attention
        attn_probs = nn.softmax(attn_logits, axis=2)
        node_feats = jnp.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(axis=2)

        if return_attn_probs:
            return node_feats, attn_probs.transpose(0, 3, 1, 2)
        
        return node_feats



def mask_tracks(x, n_trks):
    # Track-level mask
    mask = jnp.stack([
        jnp.where(jnp.array(list(range(15))) < n, 1, 0)
        # jnp.where(jnp.array(list(range(15))) < n, True, False)
        for n in n_trks
    ])
    mask = mask.reshape(*mask.shape, 1)

    r = jnp.stack([jnp.array(list(list(range(15)) for i in list(range(15))))])[0]
    l = r.T
    mask_edges = jnp.stack([
        jnp.where((r < n) & (l < n), 1, 0)
        # jnp.where((r < n) & (l < n), True, False)
        for n in n_trks
    ])  
    mask_edges = jnp.tril(mask_edges)
    return mask, mask_edges

