import optax
import jax.numpy as jnp
import sys

def relative_error(preds, y):
    return jnp.abs((preds - y) / y).mean()

def mae(preds, y):
    return jnp.abs(preds - y).mean()

def mse(preds, y):
    return optax.squared_error(preds, y).mean()

def huber(preds, y):
    return optax.huber_loss(preds, y).mean()

def get_loss_function(loss_function):

    if loss_function == 'MSE':
        return mse
    elif loss_function == 'MAE':
        return mae
    elif loss_function == 'Relative_error':
        return relative_error
    elif loss_function == 'Huber':
        return huber
    else:
        print('Unknown loss function')
        sys.exit()

