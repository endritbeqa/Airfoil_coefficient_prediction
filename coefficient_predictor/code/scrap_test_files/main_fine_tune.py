# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import jax
import os
import tensorflow as tf

from src.preprocessing.preprocess_bluff_body import generate_tfds_dataset
from naca_transformer.code.scrap_test_files.fine_tune import fine_tune
from naca_transformer.code.scrap_test_files.fine_tune_parallel import fine_tune_parallel


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
#os.environ["JAX_TRACEBACK_FILTERING"] = off

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    default='config_fine_tune.py',
    help_string='/local/disk1/ebeqa/naca_transformer/code/config.py',
    lock_config=True,
)
flags.mark_flag_as_required('config')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Hide GPUs from TF. Otherwise, TF might reserve memory and block it for JAX
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(),
                 jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    if FLAGS.config.trainer == 'preprocess':
        generate_tfds_dataset(FLAGS.config)
    elif FLAGS.config.trainer == 'fine_tune':
        fine_tune(FLAGS.config)
    elif FLAGS.config.trainer == 'fine_tune_parallel':
        fine_tune_parallel(FLAGS.config)
    elif FLAGS.config.trainer == 'inference':
        print('Implement inference')
    else:
        raise app.UsageError('Unknown trainer: {FLAGS.config.trainer}')


if __name__ == '__main__':
    jax.config.config_with_absl()
    app.run(main)
