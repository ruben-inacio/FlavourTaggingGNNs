import jax.numpy as jnp
# from jax.config import config
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", False)


def binary_cross_entropy(ytrue, ypred, weights):    
    loss = -ytrue * jnp.log(ypred) - (1. - ytrue) * jnp.log(1-ypred)
    
    if weights != None:
        loss = loss * weights
    
    return loss
    

def categorical_cross_entropy(ytrue, ypred, weights):    
    loss = -ytrue * jnp.log(ypred)
    
    if weights != None:
        loss = loss * weights
    
    return jnp.sum(loss, axis=2)


def squared_error(targets, predictions, mean=True):
    errors = (predictions - targets) ** 2
    if mean:
        return errors.mean()
    return errors


def gaussian_neg_loglikelihood(targets, predictions, errors, mean=True, eps=1e-6):
    errors = max(errors, eps)
    loss = 0.5 * (jnp.log(errors) + jnp.square(predictions - targets) / errors)

    if mean:
        return loss.mean()
    
    return loss