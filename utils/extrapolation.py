import jax
import jax.numpy as jnp     
from jax.config import config
config.update("jax_enable_x64", True)

@jax.jit
def extrapolation_not_vmapped(tracks,vertex):
    
    def phiv(rho, z, z0, phi0, theta0):
        #zR = 0
        #phiv = phi0 + (jnp.tan(theta0)/rho)*(zR + z0 - z)
        phiv = phi0 + (rho)*jnp.tan(theta0)*(z-z0)
        return phiv
        
    def xv(rho, phi, d0, phi0, theta0):
        #xR = 0
        #xv = xR + d0*jnp.cos(phi0 + jnp.pi/2.) + rho*(jnp.cos(phi + jnp.pi/2.) - jnp.cos(phi0 + jnp.pi/2.))
        L = (phi-phi0)/(rho)
        xv = d0*jnp.sin(phi0) + L*jnp.cos(phi0) - L**2 * rho/2 * jnp.sin(phi0)
        return xv
        
    def yv(rho, phi, d0, phi0, theta0):
        #yR = 0
        #yv = yR + d0*jnp.sin(phi0 + jnp.pi/2.) + rho*(jnp.sin(phi + jnp.pi/2.) - jnp.sin(phi0 + jnp.pi/2.))
        L = (phi-phi0)/(rho)
        yv = -d0*jnp.cos(phi0) + L*jnp.sin(phi0) + L**2 * rho/2 * jnp.cos(phi0)
        return yv
    
    def find_distance_3d(vtx_x, xV, vtx_y, yV, vtx_z, zV):
        all_distances = jnp.sqrt(jnp.square(vtx_x-xV) + jnp.square(vtx_y-yV) + jnp.square(vtx_z-zV))
        index_closest_approach = jnp.argmin(all_distances, axis=1)
        return index_closest_approach
    
    n_trks = 15
    z_margin = 20 #mm
    n_points = 1000
    
    varlist = [
        "trk_pt", "trk_d0", "trk_z0", "trk_phi", "trk_theta", 
        "trk_rho", "trk_ptfrac", "trk_deltar", "trk_pterr", "trk_d0err", 
        "trk_z0err", "trk_phierr", "trk_thetaerr", "trk_rhoerr", 
        "trk_signed_sigd0", "trk_signed_sigz0",
        "prod_x", "prod_y", "prod_z", 
        "hadron_x", "hadron_y", "hadron_z",
        "n_trks",   
    ]
    
    orig_d0    = tracks[:,varlist.index("trk_d0")].reshape(n_trks,1)
    orig_z0    = tracks[:,varlist.index("trk_z0")].reshape(n_trks,1)
    orig_phi   = tracks[:,varlist.index("trk_phi")].reshape(n_trks,1)
    orig_theta = tracks[:,varlist.index("trk_theta")].reshape(n_trks,1)
    orig_rho   = tracks[:,varlist.index("trk_rho")].reshape(n_trks,1)
    
    new_ref_x = jnp.repeat(vertex[0], n_trks).reshape(n_trks,1)
    new_ref_y = jnp.repeat(vertex[1], n_trks).reshape(n_trks,1)
    new_ref_z = jnp.repeat(vertex[2], n_trks).reshape(n_trks,1)
    
    extrap_z = jnp.linspace(vertex[2]-z_margin,vertex[2]+z_margin,n_points).reshape(1,n_points)
    extrap_z = jnp.repeat(extrap_z, n_trks, axis=0).reshape(n_trks,n_points)
    
    rsized_d0    = jnp.repeat(orig_d0, n_points, axis=1).reshape(n_trks,n_points)
    rsized_z0    = jnp.repeat(orig_z0, n_points, axis=1).reshape(n_trks,n_points)
    rsized_phi   = jnp.repeat(orig_phi, n_points, axis=1).reshape(n_trks,n_points)
    rsized_theta = jnp.repeat(orig_theta, n_points, axis=1).reshape(n_trks,n_points)
    rsized_rho   = jnp.repeat(orig_rho, n_points, axis=1).reshape(n_trks,n_points)
    
    extrap_phi = phiv(rsized_rho, extrap_z, rsized_z0, rsized_phi, rsized_theta)
    extrap_x = xv(rsized_rho, extrap_phi, rsized_d0, rsized_phi, rsized_theta)
    extrap_y = yv(rsized_rho, extrap_phi, rsized_d0, rsized_phi, rsized_theta)
    
    min_i = find_distance_3d(new_ref_x, extrap_x, new_ref_y, extrap_y, new_ref_z, extrap_z)
    new_p_x = jnp.array([i[j] for i,j in zip(extrap_x,min_i)]).reshape(n_trks,1)
    new_p_y = jnp.array([i[j] for i,j in zip(extrap_y,min_i)]).reshape(n_trks,1)
    new_p_z = jnp.array([i[j] for i,j in zip(extrap_z,min_i)]).reshape(n_trks,1)
    new_p_phi = jnp.array([i[j] for i,j in zip(extrap_phi,min_i)]).reshape(n_trks,1)
    new_p_theta = orig_theta
    new_p_rho = orig_rho
    
    orig_pt = tracks[:,varlist.index("trk_pt")].reshape(n_trks,1)
    orig_eta = -jnp.log(jnp.tan(orig_theta/2.))
    p0x = orig_pt * jnp.cos(new_p_phi)
    p0y = orig_pt * jnp.sin(new_p_phi)
    p0z = orig_pt * jnp.sinh(orig_eta)
    
    new_p_d0 = jnp.sqrt(jnp.square(new_p_x - new_ref_x) + jnp.square(new_p_y - new_ref_y))
    drIP = jnp.stack(((new_p_x - new_ref_x),(new_p_y - new_ref_y), jnp.zeros_like(new_ref_z)), axis=1).reshape(n_trks,3)
    p0T = jnp.stack((p0x, p0y, jnp.zeros_like(p0z)), axis=1).reshape(n_trks,3)
    cross = jnp.cross(p0T, drIP)
    sign_d0 = jnp.sign(cross[:,2]).reshape(n_trks,1)
    new_p_d0 = new_p_d0 * sign_d0
    
    new_p_z0 = new_p_z - new_ref_z
    
    new_perigee_params = jnp.concatenate((new_p_d0,new_p_z0,new_p_phi),axis = 1)
    # new_perigee_params = jnp.concatenate((new_p_x,new_p_y,new_p_z,new_p_d0,new_p_z0,new_p_phi,new_p_theta,new_p_rho,),axis = 1)
    out = jnp.concatenate([tracks[:,varlist.index("trk_pt")].reshape(-1,1), new_perigee_params, tracks[:, 4:]], axis=1)
    return out  # new_perigee_params

extrapolation_vmapped = jax.jit(jax.vmap(extrapolation_not_vmapped, in_axes=(0, 0), out_axes=(0)))

@jax.jit
def extrapolation(tracks,vertex):
    return extrapolation_vmapped(tracks,vertex)

     
    
    
    
    
    
    
    
    