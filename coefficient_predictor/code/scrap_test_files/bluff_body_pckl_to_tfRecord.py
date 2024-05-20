import pickle
import naca_transformer.code.naca_training.config as config
import tensorflow as tf
import os
import numpy as np
from src.preprocessing.conversion import create_tfExample
import random
from tqdm import tqdm
import tensorflow_datasets as tfds


def load_bluff_data(config):

    pckl_data_x_path = "/local/disk1/ebeqa/naca_transformer/bluff_body_data/data_X.pkl"
    pckl_data_y_path = "/local/disk1/ebeqa/naca_transformer/bluff_body_data/data_Y.pkl"

    f_X = open(pckl_data_x_path, 'rb')
    f_Y = open(pckl_data_y_path, 'rb')

    data_X = pickle.load(f_X)
    data_X = np.array(data_X)
    data_X = data_X.transpose([0,2,3,1])
    data_X = data_X[:,:,:,0]
    data_X = np.expand_dims(data_X, -1)
    

    data_Y = pickle.load(f_Y)
    data_Y = np.array(data_Y)
    data_Y = data_Y.transpose([0,2,3,1])
    
    print(data_X.shape)
    print(data_Y.shape)

    data_X =  data_X.tolist()
    data_Y = data_Y.tolist()

    dataset = list(zip(data_X, data_Y))

    ds_train = [dataset.pop(random.randrange(len(dataset))) for _ in 
                range(int(config.preprocess.train_split*len(dataset)))]

    ds_test = dataset

    train_quotient, train_remainder = divmod(len(ds_train), config.preprocess.nsamples)

    test_quotient, test_remainder = divmod(len(ds_test), config.preprocess.nsamples)

    n_files_train = (train_quotient + 1 if train_remainder != 0 else
                     train_quotient)
    n_files_test = (test_quotient + 1 if test_remainder != 0 else test_quotient)

    train_shards, test_shards = [], []

    for i in tqdm(range(n_files_train), desc ="Train split", position = 0):
        if train_remainder !=0 and i==n_files_train-1:
            batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in 
                     range(train_remainder)]
        else:
            batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in 
                     range(config.preprocess.nsamples)]
            

        file_dir = os.path.join(config.preprocess.writedir,
                                'bluff_body-train.tfrecord-{}-of-{}'.format(
                                    str(i).zfill(5),
                                    str(n_files_train).zfill(
                                        5)))
        with tf.io.TFRecordWriter(file_dir) as writer:
            j = 0
            for sample in tqdm(batch, desc='Shards', position=1,
                               leave=False):
        
                x, y = np.asarray(sample[0]),np.asarray(sample[1])
                example = create_tfExample(x, y, "0_0_0")
                writer.write(example.SerializeToString())

                j += 1
        train_shards.append(j)

    for i in tqdm(range(n_files_test), desc='Test split', position=0):
        if test_remainder != 0 and i == n_files_test - 1:
            batch = [ds_test.pop(random.randrange(len(ds_test))) for _ in
                     range(test_remainder)]
        else:
            batch = [ds_test.pop(random.randrange(len(ds_test))) for _ in
                     range(config.preprocess.nsamples)]

        file_dir = os.path.join(config.preprocess.writedir,
                                'bluff_body-test.tfrecord-{}-of-{}'.format(
                                    str(i).zfill(5),
                                    str(n_files_test).zfill(
                                        5)))

        with tf.io.TFRecordWriter(file_dir) as writer:
            j = 0
            for sample in tqdm(batch, desc='Shards', position=1,
                               leave=False):
                
                x, y = np.asarray(sample[0]), np.asarray(sample[1])
                example = create_tfExample(x, y, "0_0_0")
                writer.write(example.SerializeToString())

                j += 1

        test_shards.append(j)

    # Create metadata files to read dataset with tfds.load(args)
    features = tfds.features.FeaturesDict({
        'encoder': tfds.features.Tensor(
            shape=(*config.vit.img_size, 1),
            dtype=np.float32,
        ),
        'decoder': tfds.features.Tensor(
            shape=(*config.vit.img_size, 3),
            dtype=np.float32,
        ),
        'label': tfds.features.Text(
            encoder=None,
            encoder_config=None,
            doc='Simulation config: airfoil_aoa_mach'
        ),
    })

    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=train_shards,
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name='test',
            shard_lengths=test_shards,
            num_bytes=0,
        ),
    ]

    tfds.folder_dataset.write_metadata(
        data_dir=config.preprocess.writedir,
        features=features,
        split_infos=split_infos,
        filename_template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
    )


    


if __name__ == '__main__':
    
    config = config.get_config()
    load_bluff_data(config=config)



    