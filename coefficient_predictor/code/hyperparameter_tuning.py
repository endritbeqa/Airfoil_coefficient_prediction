import optuna
from train import train_and_evaluate
from config import get_config
import logging
import sys
import os


def objective(trial):
    config = get_config()
    trial_config = config.copy_and_resolve_references()

    batch_size = trial.suggest_int('batch_size', 100, 400, step=4)
    learning_rate = trial.suggest_float('learning_rate', 1e-9, 1e-3 )
    #scheduler = trial.suggest_categorical('scheduler', ['constant','sgdr'])

    directory_name = "batch_size{}_learning_rate{}".format(batch_size, learning_rate)

    trial_config.output_dir = os.path.join(trial_config.output_dir, 'trials', directory_name)

    trial_config.batch_size = batch_size
    #trial_config.learning_rate_scheduler = scheduler
    trial_config.learning_rate_end_value = learning_rate

    loss = train_and_evaluate(trial_config)

    return loss


if __name__ == '__main__':

    config = get_config()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = config.study_name  # Unique identifier of the study.
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("./journal.log"), )
    #storage_name = "sqlite:///{}.db".format(study_name)

    os.makedirs(config.output_dir, exist_ok=True)

    study = optuna.create_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=30, gc_after_trial=True, n_jobs=1)

    for plotType in ['contour', 'slice']:
        os.makedirs(os.path.join(config.output_dir, plotType), exist_ok=True)

    hyper_parameters = ['batch_size', 'learning_rate']

    for i in range(len(hyper_parameters)):
        for j in range(i + 1, len(hyper_parameters)):
            fig = optuna.visualization.plot_contour(study, params=[hyper_parameters[i], hyper_parameters[j]])
            fig.write_image(os.path.join(config.output_dir, 'contour',
                                         'contour_' + hyper_parameters[i] + '_' + hyper_parameters[j] + '.png'))

            fig = optuna.visualization.plot_slice(study, params=[hyper_parameters[i], hyper_parameters[j]])
            fig.write_image(os.path.join(config.output_dir, 'slice',
                                         'slice_' + hyper_parameters[i] + '_' + hyper_parameters[j] + '.png'))

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(config.output_dir, 'param_importance.png'))

    print(study.best_params)
