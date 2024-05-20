# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 22, 2023
# version ='1.0'
# ---------------------------------------------------------------------------


from ml_collections import ConfigDict
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator
import numpy as np




def plot_loss(config: ConfigDict, train_loss, test_loss):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x = range(0, config.num_epochs)

    ax.plot(x, train_loss, label='Train')
    ax.plot(x, test_loss, label='Test')

    ax.set_yscale('log')

    ax.set_xlim(0, config.num_epochs)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(config.loss_function)

    ax.legend(loc=1)

    plt.grid(visible=True, which='major', color='#444', linestyle='-')
    plt.grid(visible=True, which='minor', color='#ccc', linestyle='--')

    ax.set_title(
        'Epochs = {}, Batch size = {}, Lr scheduler = {}, Weight decay = {}'
        .format(config.num_epochs, config.batch_size,
                config.learning_rate_scheduler, config.weight_decay),
        fontsize=8)

    plt.savefig('{}/vit_loss.png'.format(config.output_dir),
                bbox_inches="tight",
                dpi=300)
    plt.close()



def loss_comparison(files, labels, title, hyperparameter):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    for i in range(len(files)):
        data = np.loadtxt(files[i], delimiter=',')

        ax[0].plot(range(0, data.shape[0]), data[:, 0], label=labels[i])
        ax[1].plot(range(0, data.shape[0]), data[:, 1], label=labels[i])

    for j in range(2):
        ax[j].set_yscale('log')
        ax[j].set_xlim(0, 200)
        ax[j].set_ylim(1e-6, 1e-3)
        ax[j].set_xlabel('Epochs')

        # plt.setp(ax[j].get_yticklabels(), visible=False) if j != 0 else None

        ax[j].grid(visible=True, which='major', color='#444', linestyle='-')
        ax[j].grid(visible=True, which='minor', color='#ccc',
                   linestyle='--')

    # ax[0].legend(loc=1, ncols=2, title=title)
    ax[0].set_ylabel('Train Loss (Huber $L_\delta$)')
    ax[1].set_ylabel('Test Loss (Huber $L_\delta$)')

    plt.tight_layout()

    plt.savefig('vit_loss_{}.png'.format(hyperparameter),
                bbox_inches="tight", dpi=300)
    plt.close()

