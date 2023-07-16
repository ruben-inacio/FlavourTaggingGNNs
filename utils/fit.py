import jax
import jax.numpy as jnp     
from jax.config import config
config.update("jax_enable_x64", True)

@jax.jit
def vertex_fit(tracks,weights,seed):
    
    n_trks = 16
    
    varlist = [
        "trk_pt", "trk_d0", "trk_z0", "trk_phi", "trk_theta", 
        "trk_rho", "trk_ptfrac", "trk_deltar", "trk_pterr", "trk_d0err", 
        "trk_z0err", "trk_phierr", "trk_thetaerr", "trk_rhoerr", 
        "trk_signed_sigd0", "trk_signed_sigz0",
        "prod_x", "prod_y", "prod_z", 
        "hadron_x", "hadron_y", "hadron_z",
        "n_trks",   
    ]

    def getA_B(theta,phiv,rho,Q,R):

        c = jnp.cos(phiv)
        s = jnp.sin(phiv)
        t = 1./jnp.tan(theta)
        useful_zeros = jnp.zeros(1)
        useful_ones = jnp.ones(1)

        A_1 = jnp.stack([s, -c, useful_zeros])
        A_2 = jnp.stack([-t*c, -t*s, useful_ones])
        A_3 = jnp.stack([-rho*c, -rho*s, useful_zeros])
        A_4 = jnp.stack([useful_zeros, useful_zeros, useful_zeros])
        A_5 = jnp.stack([useful_zeros, useful_zeros, useful_zeros])

        A = jnp.stack([A_1, A_2, A_3, A_4, A_5])
        
        B_1 = jnp.stack([Q, useful_zeros, -(Q**2)/2])
        B_2 = jnp.stack([-R*t, Q*(1+t**2), Q*R*t])
        B_3 = jnp.stack([useful_ones, useful_zeros, -Q])
        B_4 = jnp.stack([useful_zeros, useful_ones, useful_zeros])
        B_5 = jnp.stack([useful_zeros, useful_zeros, useful_ones])

        B = jnp.stack([B_1, B_2, B_3, B_4, B_5])

        return A, B
    
    def get_qmeas(track):

        d0     = track[varlist.index("trk_d0")]
        z0     = track[varlist.index("trk_z0")]
        phi    = track[varlist.index("trk_phi")]
        theta  = track[varlist.index("trk_theta")]
        rho    = track[varlist.index("trk_rho")]
        q = jnp.stack([d0, z0, phi, theta, rho])    
        q = jnp.reshape(q,(5,1))

        return q
    
    def get_qpred(rv, pv):
        
        phiv = pv[0]
        theta = pv[1]
        rho = pv[2]
        
        Q = rv[0]*jnp.cos(phiv) + rv[1]*jnp.sin(phiv)
        R = rv[1]*jnp.cos(phiv) - rv[0]*jnp.sin(phiv)
        
        h1 = -R - Q**2 * (rho)/2
        h2 = rv[2] - Q*(1-R*(rho))/jnp.tan(theta)
        h3 = phiv - Q*(rho)
        h4 = theta
        h5 = rho
        h = jnp.stack([h1,h2,h3,h4,h5])
        h = jnp.reshape(h,(5,1))
        
        return h
    
    def get_cov(track):

        ind = varlist.index("trk_d0err")
        track_err = jnp.array(track[ind:ind+5])

        return abs(track_err)

    def get_per_track(rv, pv, track):

        qmeas = get_qmeas(track)

        phiv = pv[0]
        theta = pv[1]
        rho = pv[2]
        
        Q = rv[0]*jnp.cos(phiv) + rv[1]*jnp.sin(phiv)
        R = rv[1]*jnp.cos(phiv) - rv[0]*jnp.sin(phiv)

        A, B = getA_B(theta,phiv,rho,Q,R)
        A = jnp.squeeze(A)
        B = jnp.squeeze(B)
        
        h = get_qpred(rv, pv)
        
        rv = jnp.reshape(rv,(3,1))
        pv = jnp.reshape(pv,(3,1))
        
        G = jnp.diag(1./get_cov(track)**2)
        Di = jnp.transpose(A) @ G @ B
        D0 = jnp.transpose(A) @ G @ A
        E = jnp.transpose(B) @ G @ B 
        W = jnp.linalg.pinv(E)
        q_c = qmeas - (h - A @ rv - B @ pv)
        
        return A,B,Di,D0,G,W,q_c
    
    def make_estimate(point, mom, weight):
        
        def per_track_v_estimate(i, params):
            v, cov = params
            A,B,Di,D0,G,W,q_c = get_per_track(point, mom[i], tracks[i])
            v = v + (jnp.transpose(A) @ G @ (jnp.eye(5) - B @ W @ jnp.transpose(B) @ G) @ q_c) * weight[i]
            cov = cov + (D0 - Di @ W @ jnp.transpose(Di)) * weight[i]
            return (v,cov)
        
        params = jax.lax.fori_loop(0, n_trks, per_track_v_estimate, (jnp.zeros((3,1), dtype=jnp.float64), jnp.zeros((3,3), dtype=jnp.float64)))
        vn_wo_Cn, Cn_inv = params
        
        Cn = jnp.linalg.pinv(Cn_inv)
        vn = Cn @ vn_wo_Cn   
        
        mn = jnp.zeros((n_trks,3,1), dtype=jnp.float64)
        def per_track_p_estimate(i, params):
            m = params
            A,B,Di,D0,G,W,q_c = get_per_track(point, mom[i], tracks[i])
            mi = W @ jnp.transpose(B) @ G @ (q_c - A @ vn)
            m = m.at[i].set(mi)
            return m
            
        mn = jax.lax.fori_loop(0, n_trks, per_track_p_estimate, mn)  
        
        return jnp.reshape(vn,(3,)), mn, Cn
    
    def estimate_vertex(weight):
    
        v = seed
        cov = jnp.zeros((3,3), dtype=jnp.float64)
        p = jnp.zeros((n_trks,3,1), dtype=jnp.float64)
        for i in range(n_trks):
            phi0   = jnp.array([tracks[i][varlist.index("trk_phi")]])
            theta  = jnp.array([tracks[i][varlist.index("trk_theta")]])
            rho    = jnp.array([tracks[i][varlist.index("trk_rho")]])

            pv = jnp.stack([phi0, theta, rho])
            pv = jnp.reshape(pv,(3,1))
            p = p.at[i].set(pv)

        for i in range(10):
            v, p, cov = jax.lax.stop_gradient(make_estimate(v, p, weight))
            
        vp = jnp.concatenate((v, jnp.ravel(p))).reshape(-1)
        
        def chi2(fitvars,we):
            
            r = fitvars[0:3]
            p = fitvars[3:].reshape(n_trks,3)
        
            t = tracks
            w = we
            x = 0

            for i in range(n_trks):

                qmeas = get_qmeas(t[i])
                qpred = get_qpred(r, p[i])
                G = jnp.diag((1./get_cov(t[i])**2))
                dq = qmeas - qpred
                x = x + (jnp.transpose(dq) @ G @ dq) * w[i]

            return x
        
        x2 = chi2(vp,weight)
        
        x2_hess = jax.hessian(chi2, argnums=(0,1), has_aux=False)(vp,weight)        
        grad_v = -jnp.linalg.inv(x2_hess[0][0]) @ x2_hess[0][1]
        grad_v = jax.numpy.nan_to_num(grad_v, nan=0., posinf=1e200, neginf=-1e200)
        
        return v, cov, grad_v, x2
    
    v, cov, grad_v, x2 = estimate_vertex(weights)
    
    grad_v = grad_v.reshape(3+3*n_trks,n_trks)
    grad_v = grad_v[0:3,:]
    
    return v, cov, x2, grad_v

vertex_fit_vmap = jax.jit(jax.vmap(vertex_fit, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0)))

def custom(f):
    
    n_trks = 16
    
    @jax.custom_vjp
    def custom_f(*args):
        return f(*args)[0:3]
    
    def custom_fwd(*args):
        return f(*args)[0:3], args
    
    def custom_bwd(res, dy):
        t, w, s = res
        grad_v = f(t,w,s)[3]
        
        grad_output_v = jnp.reshape(dy[0], (-1, 1, 3))
        grad_v = jnp.reshape(grad_v, (-1, 3, n_trks))
        
        batch_grads_v = jnp.einsum('bij,bjk->bik', grad_output_v, grad_v).reshape(-1, n_trks)
        batch_grads_w = batch_grads_v
        
        return None, batch_grads_w, None
    
    custom_f.defvjp(custom_fwd, custom_bwd)
    
    return custom_f

ndive_layer = custom(vertex_fit_vmap)

@jax.jit
def ndive(tracks, weights, seed):
    return ndive_layer(tracks, weights, seed)