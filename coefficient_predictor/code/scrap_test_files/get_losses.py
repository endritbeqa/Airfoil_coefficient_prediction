from jax.tree_util import tree_structure
from flax.training import train_state, orbax_utils, common_utils
import jax
import orbax
import jax.numpy as jnp
from flax import traverse_util
import optax
from ml_collections import ConfigDict, FrozenConfigDict
import tensorflow_datasets as tfds
import tensorflow as tf
from jax.random import PRNGKey
import logging
import os
from flax import traverse_util
import numpy as np
from typing import Tuple, Any

from src.transformer.network import VisionTransformer
from src.utilities.schedulers import load_learning_rate_scheduler
from src.utilities.visualisation import plot_predictions, plot_delta, plot_loss, plot_fields
from src.utilities.pressure_preprocesing import *
from fine_tune import config_fine_tune as config_file


import numpy as np
import matplotlib.pyplot as plt

def get_fine_tune_data_from_tfds(*, config, mode):
    builder = tfds.builder_from_directory(builder_dir=config.dataset)

    ds = builder.as_dataset(
        split=tfds.split_for_jax_process(mode)
    )

    if mode == 'train':
        ds = ds.shuffle(config.shuffle_buffer_size,
                        seed=0,
                        reshuffle_each_iteration= True)

        ds.repeat(config.num_epochs)

    ds = ds.batch(batch_size= config.batch_size, 
                  drop_remainder = True,
                  num_parallel_calls = tf.data.AUTOTUNE
                  ) 

    return ds.prefetch(tf.data.AUTOTUNE)




#@jax.jit
def train_step(state: train_state.TrainState, batch: jnp.ndarray,
               rng: PRNGKey) -> Tuple[train_state.TrainState, Any]:
    # Generate new dropout key for each step
    rng_dropout = jax.random.fold_in(rng, state.step)

    

    def loss_fn(params):
        preds = state.apply_fn({'params': params},
                               batch['encoder'], batch['decoder'],
                               train=True,
                               rngs={'dropout': rng_dropout},
                               )
        loss = optax.huber_loss(preds, batch['decoder']).mean()
        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)

    filename = str(batch['label'][0])
    with open('/local/disk1/ebeqa/naca_transformer/outputs/losses/losses_training.txt' , 'a') as file:
            file.write(filename)
            file.write(':'+str(loss))
            file.write('\n')

    return state, loss

def test_step(state: train_state.TrainState, batch: jnp.ndarray):
    preds = state.apply_fn({'params': state.params},
                           batch['encoder'], batch['decoder'],
                           train=False,
                           )

    loss = optax.huber_loss(preds, batch['decoder']).mean()
    
    filename = str(batch['label'][0])
    with open('/local/disk1/ebeqa/naca_transformer/outputs/losses/losses_testing.txt' , 'a') as file:
            file.write(filename)
            file.write(':'+str(loss))
            file.write('\n')
        

    return preds, loss


def load_model_from_checkpoint(config: ConfigDict):
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    variables_restored = orbax_checkpointer.restore(config.checkpoint_dir)

    return variables_restored


def create_train_state(params_key: PRNGKey, config: ConfigDict, lr_scheduler):
    # Create model instance
    model = VisionTransformer(config.vit)

    # Initialise model and use JIT to reside params in CPU memory
    ckpt = load_model_from_checkpoint(config=config)
    restored_model = ckpt['model']
    variables = restored_model['params']
    
    # Initialise train state
    tx_trainable = optax.adamw(learning_rate=lr_scheduler,
                     weight_decay=config.weight_decay)

    tx_frozen = optax.set_to_zero()

    partition_optimizers = {'trainable': tx_trainable, 'frozen': tx_frozen}

    trainable_layers = config.layers_to_train

    param_partitions = traverse_util.path_aware_map(
        lambda path, v: 'trainable' if any(layer in path for layer in trainable_layers) else 'trainable', variables)


    tx = optax.multi_transform(partition_optimizers, param_partitions)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=tx
    )

def fine_tune(config: ConfigDict):

    os.makedirs(config.output_dir, exist_ok=True)
    logging.info("Initialising fine tune dataset.")

    ds_train = get_fine_tune_data_from_tfds(config=config, mode='train')
    ds_test = get_fine_tune_data_from_tfds(config=config, mode='test')

    print(ds_train.cardinality().numpy())
    print(ds_test.cardinality().numpy())

    total_steps = ds_train.cardinality().numpy()

    # Create PRNG key
    rng = jax.random.PRNGKey(0)
    # Split PRNG key into required keys
    rng, rng_params, rng_dropout = jax.random.split(rng, num=3)

    # Create learning rate scheduler
    lr_scheduler = load_learning_rate_scheduler(
        config=config, name=config.learning_rate_scheduler,
        total_steps=total_steps)

    # Create TrainState
    state = create_train_state(rng_params, config, lr_scheduler)
    #state = load_model_from_checkpoint(config=config)
    train_log, test_log = [], []

    # Generate index array to plot n samples from the test data
    rng = np.random.default_rng(0)

    logging.info("Starting training loop. Initial compile might take a while.")
            
    for test_batch in tfds.as_numpy(ds_test):
                
        preds, test_loss = test_step(state, test_batch)
        test_log.append(test_loss)

    with open('/local/disk1/ebeqa/naca_transformer/outputs/losses/losses_testing.txt' , 'a') as file:
            
        file.write('went through testing')
        file.write('\n')       
           


if __name__=='__main__':
     config = config_file.get_config()

     fine_tune(config)