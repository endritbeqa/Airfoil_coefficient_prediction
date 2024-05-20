import os

import jax
from absl import logging
import orbax
from flax.training import train_state
from ml_collections import ConfigDict
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any
import jax.numpy as jnp
import numpy as np

from Airfoil_coefficient_prediction.coefficient_predictor.code.src.transformer.network import VisionTransformer
from src.transformer.input_pipeline import get_data_from_tfds


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# Hide GPUs from TF. Otherwise, TF might reserve memory and block it for JAX
tf.config.experimental.set_visible_devices([], 'GPU')

PRNGKey = Any

from PIL import Image
import numpy as np


def array_to_grayscale_image(array, save_path=None):
    """
    Convert an array into a grayscale image.

    Parameters:
        array (numpy.ndarray): The input array representing the image.
        save_path (str, optional): The file path to save the image. If None, the image will be displayed instead of being saved.

    Returns:
        None
    """
    # Ensure the array is a 2D or 3D array (e.g. single-channel or RGB image)
    if array.ndim not in [2, 3]:
        raise ValueError("Input array should be a 2D (grayscale) or 3D (RGB) array.")

    # Convert the array into a Pillow Image object
    if array.ndim == 3:
        # Convert a 3D array (RGB) to grayscale using the 'L' mode (luminance)
        image = Image.fromarray(array, 'RGB').convert('L')
    else:
        # Convert a 2D array (already grayscale) directly to an image
        image = Image.fromarray(array, 'L')

    # Save or display the image
    if save_path:
        image.save(save_path)
        print(f"Image saved at {save_path}")
    else:
        image.show()


def convert_to_255_range(arr):
    """
    Convert an array of numerical values to the range [0, 255].

    Args:
    arr (list or numpy array): Array of numerical values.

    Returns:
    list or numpy array: Array with values in the range [0, 255].
    """
    import numpy as np

    # Convert array to numpy array if it's not already
    arr = np.array(arr)

    # Find the minimum and maximum values in the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalize the array to the range [0, 1]
    if max_val != min_val:  # Check to avoid division by zero
        normalized_arr = (arr - min_val) / (max_val - min_val)
    else:
        normalized_arr = arr - min_val

    # Scale the normalized array to the range [0, 255]
    scaled_arr = normalized_arr * 255

    # Return the scaled array
    return scaled_arr.astype(np.uint8)



def load_model(config: ConfigDict):
    rng = jax.random.PRNGKey(0)

    model = VisionTransformer(config)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(config.checkpoint_dir)
    params = ckpt['model']['params']

    variables = jax.jit(lambda: model.init(rng, jnp.ones(
        [config.batch_size, *config.vit.img_size, 1]), train=False),
                        )()

    variables['params'] = params

    tx = optax.set_to_zero()

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )



def evaluate(config: ConfigDict):

    logging.info("Initialising dataset.")
    os.makedirs(config.output_dir, exist_ok=True)

    ds_test = get_data_from_tfds(config=config, mode='test')

    state = load_model(config)

    loss_log = []
    result_log = ''
    count = 1

    for batch in tfds.as_numpy(ds_test):
        x = batch['encoder']
        x_pic = x.squeeze()
        x_pic = convert_to_255_range(x_pic)
        array_to_grayscale_image(x_pic,'{}/{}.png'.format(config.output_dir, count))
        count+=1

        preds = state.apply_fn({'params': state.params}, x, train=False)

        targets = []

        for entry in batch['label']:
            entry = entry.decode("utf-8")
            label_data = entry.split('_')
            coefficients = [float(coefficient) for coefficient in label_data[3:]]
            targets.append(coefficients)

        targets = np.array(targets)
        sample_loss = optax.squared_error(preds, targets ).mean()
        loss_log.append(sample_loss)
        for i, prediction in enumerate(preds):
            target_coefficients = targets[i]
            result_log = result_log + '{},{},{},{},{},{} \n'.format(prediction[0],prediction[1],prediction[2], target_coefficients[0], target_coefficients[1], target_coefficients[2])

    print(np.array(loss_log).mean())

    with open('{}/test_predictions.csv'.format(config.output_dir), 'w') as file:
        file.write(result_log)

