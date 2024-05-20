import numpy as np
from config import get_config


if __name__ == '__main__':
    config = get_config()
    data = np.genfromtxt('{}/test_predictions.csv'.format(config.output_dir), delimiter=',')
    preds, target = data[:, 0:3], data [:, 3:6]
    relative_error = np.abs((preds-target)/target)
    relative_error = np.sort(relative_error, axis=0)
    means = np.mean(relative_error,axis=0)
    for mean in means:
        print(str(mean)+'\n')
