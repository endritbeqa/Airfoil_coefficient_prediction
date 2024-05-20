from jax.tree_util import tree_structure
from flax.training import train_state, orbax_utils, common_utils
import orbax
import os
from functools import partial
import optax
from ml_collections import ConfigDict, FrozenConfigDict
import tensorflow_datasets as tfds
from jax.random import PRNGKey
import logging

from src.transformer.input_pipeline import get_data_from_tfds
from src.transformer.network import VisionTransformer
from src.utilities.schedulers import load_learning_rate_scheduler
from src.utilities.visualisation import plot_delta, plot_loss, plot_fields
from src.utilities.pressure_preprocesing import *

@partial(jax.pmap, axis_name='num_devices')
def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray,
               key: PRNGKey):
    # Generate new dropout key for each step
    dropout_key = jax.random.fold_in(key, state.step)

    def loss_fn(params):
        preds = state.apply_fn({'params': params}, x, y, train=True,
                               rngs={'dropout': dropout_key},
                               )

        loss = optax.squared_error(preds, y).mean()

        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)

    # Combine gradients and loss across devices
    loss = jax.lax.pmean(loss, axis_name='num_devices')
    grads = jax.lax.pmean(grads, axis_name='num_devices')
  

    # Synchronise state across devices with averaged gradient
    state = state.apply_gradients(grads=grads)
    return state, loss



@partial(jax.pmap, axis_name='num_devices')
def test_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
    preds = state.apply_fn({'params': state.params}, x, y, train=False)

    loss = optax.squared_error(preds, y).mean()
    loss = jax.lax.pmean(loss, axis_name='num_devices')
    
    return preds, loss


def load_model_from_checkpoint(config: ConfigDict):
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    variables_restored = orbax_checkpointer.restore(config.checkpoint_dir)

    return variables_restored

@partial(jax.pmap, static_broadcasted_argnums=(2, 3,))
def create_train_state(params_key: PRNGKey,variables, config: ConfigDict, lr_scheduler):
    # Create model instance
    model = VisionTransformer(config.vit)

    # Initialise train state
    tx = optax.adamw(learning_rate=lr_scheduler,
                     weight_decay=config.weight_decay)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=tx
    )


def fine_tune_parallel(config : ConfigDict):

    os.makedirs(config.output_dir, exist_ok=True)
    logging.info("Initialising fine tune dataset.")

    ds_train = get_data_from_tfds(config=config, mode = 'train')
    ds_test = get_data_from_tfds(config = config, mode = 'test') 

    steps_per_epoch = ds_train.cardinality().numpy() / config.num_epochs
    total_steps = ds_train.cardinality().numpy()

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, jax.local_device_count())
    rng_idx = np.random.default_rng(0)
    sample_idx = rng_idx.integers(0, config.batch_size/4, 10)

    lr_scheduler = load_learning_rate_scheduler(
        config=config, name=config.learning_rate_scheduler,
        total_steps=total_steps
    )

    ckpt = load_model_from_checkpoint(config=config)
    restored_model = ckpt['model']
    params = restored_model['params']

    n_devices = jax.local_device_count()
    replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)

    state = create_train_state(rng, replicated_params, FrozenConfigDict(config), lr_scheduler)

    train_metrics, test_metrics, train_log, test_log = [], [], [], []

    logging.info("Starting training loop. Initial compile might take a while.")
    for step, batch in enumerate(tfds.as_numpy(ds_train)):
        
        if config.internal_geometry.set_internal_value:
            batch = set_geometry_internal_value(batch, config.internal_geometry.value)

        if config.pressure_preprocessing.enable:
            batch = pressure_preprocessing(batch, config)
        
        x_train = common_utils.shard(batch['encoder'])
        y_train = common_utils.shard(batch['decoder'])
        
        batch.pop('label')

        state, train_loss = train_step(state, x_train, y_train, rng)
        train_log.append(train_loss)

        if (step + 1) % int(steps_per_epoch) == 0 and step != 0:
            epoch = int((step + 1) / int(steps_per_epoch))

            for test_batch in tfds.as_numpy(ds_test):
                
                if config.internal_geometry.set_internal_value:
                    test_batch = set_geometry_internal_value(test_batch, config.internal_geometry.value)

                if config.pressure_preprocessing.enable:
                    test_batch = pressure_preprocessing(test_batch, config)

                x_test = common_utils.shard(test_batch['encoder'])
                y_test = common_utils.shard(test_batch['decoder'])
                
                test_batch.pop('label')

                preds, test_loss = test_step(state, x_test, y_test)
                test_log.append(test_loss)

            train_loss = np.mean(train_log)
            test_loss = np.mean(test_log)

            train_metrics.append(train_loss)
            test_metrics.append(test_loss)
            

            logging.info(
                'Epoch {}: Train_loss = {}, Test_loss = {}'.format(
                    epoch, train_loss, test_loss))
                        

            # Reset epoch losses
            train_log.clear()
            test_log.clear()

            if epoch % config.output_frequency == 0:
                for i in sample_idx:
                    pred_data = preds[0,i,:,:,:].squeeze()
                    test_data = y_test[0,i,:,:,:].squeeze()
                    plot_delta(config, pred_data, test_data, epoch, i)
                    plot_fields(config, pred_data, test_data, epoch, i)
            if epoch == config.num_epochs:
                for i in sample_idx:
                    pred_data = preds[0,i,:,:,:].squeeze()
                    test_data = y_test[0,i,:,:,:].squeeze()
                    plot_delta(config, pred_data, test_data, epoch, i)
                    plot_fields(config, pred_data, test_data, epoch, i)


    # summary_writer.flush()

    # Data analysis plots
    try:
        plot_loss(config, train_metrics, test_metrics)
    except ValueError:
        pass

    # save raw loss data into txt-file
    raw_loss = np.concatenate((train_metrics, test_metrics))
    raw_loss = raw_loss.reshape(2, -1).transpose()
    np.savetxt('{}/loss_raw.txt'.format(config.output_dir), raw_loss,
               delimiter=',')

    # Save model
    # TODO multi-process checkpointing

    print(tree_structure(state))

    ckpt = {'model': state.params}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('{}/nacaVIT'.format(config.output_dir), ckpt,
                            save_args=save_args)
