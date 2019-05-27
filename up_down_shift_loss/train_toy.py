#!/usr/bin/env python

import logging

logger = logging.getLogger("up_down_training")

import argparse
import yaml
import os
import pickle
import numpy as np

import models


def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(
        description="Train pivot adversarial network")
    parser.add_argument("config", help="Path to training config file")
    parser.add_argument("fold", type=int, help="Select the fold to be trained")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Parse config.")
    return yaml.load(open(filename, "r"))


def setup_logging(level, output_file=None):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not output_file == None:
        file_handler = logging.FileHandler(output_file, "w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def draw_plots(variable_name, history, y_label):

    # NOTE: Matplotlib needs to be imported after Keras/TensorFlow because of conflicting libraries
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.clf()

    epochs = range(1, len(history.history[variable_name]) + 1)
    plt.plot(epochs, history.history[variable_name], lw=3, label="Train {}".format(variable_name))
    plt.plot(
        epochs, history.history["val_{}".format(variable_name)], lw=3, label="Validation {}".format(variable_name))
    plt.xlabel("Epoch"), plt.ylabel(y_label)
    path_plot = os.path.join(config["output_path"],
                             "fold{}_{}".format(args.fold, variable_name))
    plt.legend()
    plt.savefig(path_plot+".png", bbox_inches="tight")
    plt.savefig(path_plot+".pdf", bbox_inches="tight")


def draw_validation_losses(variable_names, history, y_label):
    # NOTE: Matplotlib needs to be imported after Keras/TensorFlow because of conflicting libraries
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.clf()
    for variable_name in variable_names:
        epochs = range(1, len(history.history['loss_{}'.format(variable_name)]) + 1)
        plt.plot(
            epochs, history.history["val_loss_{}".format(variable_name)], lw=3, label="Validation {}".format(variable_name))
    plt.xlabel("Epoch"), plt.ylabel(y_label)
    path_plot = os.path.join(config["output_path"],
                             "fold{}_{}_variables".format(args.fold, 'validation_loss'))
    plt.legend()
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")

def draw_training_losses(variable_names, history, y_label):
    # NOTE: Matplotlib needs to be imported after Keras/TensorFlow because of conflicting libraries
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.clf()
    for variable_name in variable_names:
        epochs = range(1, len(history.history['loss_{}'.format(variable_name)]) + 1)
        plt.plot(
            epochs, history.history["loss_{}".format(variable_name)], lw=3, label="Train {}".format(variable_name))
    plt.xlabel("Epoch"), plt.ylabel(y_label)
    path_plot = os.path.join(config["output_path"],
                             "fold{}_{}".format(args.fold, 'train_loss'))
    plt.legend()
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")


def main(args, config):
    # Set seed and import packages
    # NOTE: This need to be done before any keras module is imported!
    logger.debug("Import packages and set random seed to %s.",
                 int(config["seed"]))

    np.random.seed(int(config["seed"]))

    n_samples = config['n_samples']
    mean = config['uncertainty_mean']

    # Make sure output path exists
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load training dataset

    filename = config["datasets"][args.fold].format(n_samples, mean)
    logger.debug("Load dataset from %s.", filename)
    with open(filename, "rb") as fd:
        X, y, w_train_up, w_train_down = pickle.load(fd)

    from sklearn.model_selection import train_test_split

    X_train, x_val, y_train, y_val, weights_up_train, weights_up_val, weights_down_train, weights_down_val = train_test_split(
        X, y, w_train_up, w_train_down, test_size=0.25, random_state=1234)

    # Train model
    logger.info("Train keras model %s.", config["model"]["name"])

    if config["model"]["batch_size"] < 0:
        batch_size = X_train.shape[0]
    else:
        batch_size = config["model"]["batch_size"]

    # Add callbacks
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
    callbacks = []
    if "early_stopping" in config["model"]:
        logger.info("Stop early after %s tries.",
                    config["model"]["early_stopping"])
        callbacks.append(
            EarlyStopping(patience=config["model"]["early_stopping"]))

    path_model = os.path.join(config["output_path"],
                              "toy_model.h5".format(args.fold))
    if "save_best_only" in config["model"]:
        if config["model"]["save_best_only"]:
            logger.info("Write best model to %s.", path_model)
            callbacks.append(
                ModelCheckpoint(path_model, save_best_only=True, verbose=1))

    if "reduce_lr_on_plateau" in config["model"]:
        logger.info("Reduce learning-rate after %s tries.",
                    config["model"]["reduce_lr_on_plateau"])
        callbacks.append(
            ReduceLROnPlateau(
                patience=config["model"]["reduce_lr_on_plateau"], verbose=1))

    model_impl = getattr(models, config["model"]["name"])

    output_names = ['ce','up', 'down']

    model = model_impl(X_train.shape[1], 1)

    history = model.fit(
        [X_train, weights_up_train, weights_down_train],
        y_train,
        validation_data=([x_val, weights_up_val, weights_down_val], y_val),
        batch_size=batch_size,
        epochs=config["model"]["epochs"],
        shuffle=True,
        callbacks=callbacks)

    # Plot metrics

    draw_plots(variable_name="loss", history=history, y_label="Loss")

    draw_validation_losses(output_names, history, y_label='Loss')
    draw_training_losses(output_names, history, y_label='Loss')

    # Save model

    #logger.info("Write model to {}.".format(output_path))
    #model.save(output_path + "/toy_model.h5", args.fold)

if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
