from config import get_config
from src.transformer.input_pipeline import get_data_from_tfds
from src.utilities.pressure_preprocesing import *
import tensorflow_datasets as tfds
import jax.numpy as jnp


def get_mean_std(decoder_input, geometry_internal_value):
    pressure_field_copy = jnp.copy(decoder_input[:, :, 0])

    pressure_field_copy = jnp.where(pressure_field_copy == geometry_internal_value, jnp.nan, pressure_field_copy)

    mean = jnp.nanmean(pressure_field_copy)
    std_deviation = jnp.nanstd(pressure_field_copy)
    return  mean ,std_deviation



def fun():
    config_file  =  get_config()

    ds_test = get_data_from_tfds(config = config_file, mode = 'test')
    geometry_internal_value = config_file.internal_geometry.value


    for batch in tfds.as_numpy(ds_test):
        label = batch['label'][0]

        sample = batch['decoder'][0]

        mean, std = get_mean_std(sample, geometry_internal_value)

        with open('/local/disk1/ebeqa/naca_transformer/outputs/mean_std.txt', 'a') as file:
            text = str(label)+','+str(mean)+','+str(std)+'\n'
            file.write(text) 


if __name__ == '__main__':
    fun()


