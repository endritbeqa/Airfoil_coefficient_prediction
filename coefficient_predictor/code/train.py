import os
import jax.debug
from absl import logging
from flax import traverse_util
from flax.training import train_state, orbax_utils
import orbax
from ml_collections import ConfigDict, FrozenConfigDict
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any
import numpy as np
import jax.numpy as jnp
from jax.tree_util import Partial

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from src.transformer.input_pipeline import get_data_from_tfds
from src.transformer.network import VisionTransformer
from src.utilities.schedulers import load_learning_rate_scheduler
from src.utilities.visualisation import plot_loss
from src.utilities.loss_functions import get_loss_function


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# Hide GPUs from TF. Otherwise, TF might reserve memory and block it for JAX
tf.config.experimental.set_visible_devices([], 'GPU')




PRNGKey = Any


def create_train_state(params_key: PRNGKey, config: ConfigDict, lr_scheduler):

    if config.fine_tune.enable:

        if config.fine_tune.load_train_state:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            ckpt = orbax_checkpointer.restore(config.fine_tune.checkpoint_dir)
            return ckpt['model']

        model = VisionTransformer(config)

        # Initialise model and use JIT to reside params in CPU memory
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(config.fine_tune.checkpoint_dir)
        restored_model = ckpt['model']
        restored_variables = restored_model['params']
        encoder = restored_variables['Encoder']

        variables = jax.jit(lambda: model.init(params_key, jnp.ones(
            [config.batch_size, *config.vit.img_size, 1]), train=False),
                            )()

        params = variables['params']
        params['Encoder'] = encoder


        # Initialise train state
        tx_trainable = optax.adamw(learning_rate=lr_scheduler,
                                   weight_decay=config.weight_decay)

        tx_frozen = optax.set_to_zero()

        partition_optimizers = {'trainable': tx_trainable, 'frozen': tx_frozen}

        trainable_layers = config.fine_tune.layers_to_train

        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'trainable' if any(layer in path for layer in trainable_layers) else 'frozen', params)

        tx = optax.multi_transform(partition_optimizers, param_partitions)

        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )

    # if you want to continue training from a saved Train State
    if config.load_train_state:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(config.checkpoint_dir)
        return ckpt['model']

    # Create model instance
    model = VisionTransformer(config)

    # Initialise model and use JIT to reside params in GPU memory
    variables = jax.jit(lambda: model.init(params_key, jnp.ones(
        [config.batch_size, *config.vit.img_size, 1]), train=False),
                        )()

    # Initialise train state
    tx = optax.adamw(learning_rate=lr_scheduler,
                     weight_decay=config.weight_decay)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

@jax.jit
def train_step(state: train_state.TrainState, loss_function: Partial,x: jnp.ndarray, y: jnp.ndarray,
               key: PRNGKey):
    # Generate new dropout key for each step
    dropout_key = jax.random.fold_in(key, state.step)

    def loss_fn(params):
        preds = state.apply_fn({'params': params}, x, train=True,
                               rngs={'dropout': dropout_key},
                               )

        loss = loss_function(preds, y)

        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss

@jax.jit
def test_step(state: train_state.TrainState, loss_function: Partial, x: jnp.ndarray, y: jnp.ndarray):
    preds = state.apply_fn({'params': state.params}, x, train=False)

    loss = loss_function(preds, y)

    return preds, loss


def train_and_evaluate(config: ConfigDict):

    if config.train_parallel:
        num_devices = len(jax.local_devices())
        devices = mesh_utils.create_device_mesh((num_devices,))
        input_sharding = PositionalSharding(devices).reshape(num_devices, 1, 1, 1)
        target_sharding = PositionalSharding(devices).reshape(num_devices, 1)

    logging.info("Initialising dataset.")
    os.makedirs(config.output_dir, exist_ok=True)

    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_validation = get_data_from_tfds(config=config, mode='validation')

    steps_per_epoch = ds_train.cardinality().numpy() / config.num_epochs
    total_steps = ds_train.cardinality().numpy()

    # Create PRNG key
    rng = jax.random.PRNGKey(0)
    rng_state = jax.random.PRNGKey(0)

    # Create learning rate scheduler
    lr_scheduler = load_learning_rate_scheduler(
        config=config, name=config.learning_rate_scheduler,
        total_steps=total_steps
    )

    state = create_train_state(rng_state, FrozenConfigDict(config), lr_scheduler)
    loss_function = get_loss_function(config.loss_function)
    loss_function  = jax.tree_util.Partial(loss_function)

    if config.train_parallel:
        state = jax.device_put(state, input_sharding.replicate())

    train_metrics, validation_metrics, train_log, validation_log = [], [], [], []

    logging.info("Starting training loop. Initial compile might take a while.")
    for step, batch in enumerate(tfds.as_numpy(ds_train)):

        y_train = []

        for entry in batch['label']:
            entry = entry.decode("utf-8")
            label_data = entry.split('_')
            coefficients = [float(coefficient) for coefficient in label_data[3:]]
            y_train.append(coefficients)

        x_train = batch['encoder']
        y_train = jnp.array(y_train)

        if config.train_parallel:
            x_train = jax.device_put(x_train,input_sharding)
            y_train = jax.device_put(y_train, target_sharding)

        state, train_loss = train_step(state, loss_function,x_train, y_train, rng)
        train_log.append(train_loss)

        if (step + 1) % int(steps_per_epoch) == 0 and step != 0:
            epoch = int((step + 1) / int(steps_per_epoch))

            for validation_batch in tfds.as_numpy(ds_validation):

                y_validation = []

                for entry in validation_batch['label']:
                    entry = entry.decode("utf-8")
                    label_data = entry.split('_')
                    coefficients = [float(coefficient) for coefficient in label_data[3:]]
                    y_validation.append(coefficients)


                x_validation = validation_batch['encoder']
                y_validation = jnp.array(y_validation)

                if config.train_parallel:
                    x_validation = jax.device_put(x_validation,input_sharding)
                    y_validation = jax.device_put(y_validation,target_sharding)

                predictions, validation_loss = test_step(state, loss_function,x_validation, y_validation)
                validation_log.append(validation_loss)


            train_loss = np.mean(train_log)
            validation_loss = np.mean(validation_log)

            train_metrics.append(train_loss)
            validation_metrics.append(validation_loss)
            

            logging.info(
                'Epoch {}: Train_loss = {}, Test_loss = {}'.format(
                    epoch, train_loss, validation_loss))

            # Reset epoch losses
            train_log.clear()
            validation_log.clear()

            if epoch % config.output_frequency == 0:

                ckpt = {'model': state}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save('{}/checkpoints/{}'.format(config.output_dir, str(epoch)), ckpt,
                                        save_args=save_args)


    # Data analysis plots
    try:
        plot_loss(config, train_metrics, validation_metrics)
    except ValueError:
        pass

    # save raw loss data into txt-file
    raw_loss = np.concatenate((train_metrics, validation_metrics))
    raw_loss = raw_loss.reshape(2, -1).transpose()
    np.savetxt('{}/loss_raw.txt'.format(config.output_dir), raw_loss,
               delimiter=',')

    # write config file to check setup later if necessary
    config_dir ='{}/config'.format(config.output_dir)
    os.makedirs(config_dir, exist_ok=True)
    config_filepath = os.path.join(config_dir, 'config.txt')
    with open(config_filepath, "w") as outfile:
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            outfile.write('%s:%s\n'%(key,value))

    ckpt = {'model': state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('{}/checkpoints/{}'.format(config.output_dir, 'Final'), ckpt,
                            save_args=save_args)

    jax.clear_caches()
    return validation_metrics[-1]

    

