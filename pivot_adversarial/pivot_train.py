#!/usr/bin/env python

import logging
logger = logging.getLogger("pivot_training")

import argparse
import yaml
import os
import pickle
import numpy as np

import pivot_models

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

def plot_losses(losses, save_path, lam):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["font.size"] = 16.0
    ax1 = plt.subplot(311)
    values = np.array(losses["L_f"])
    plt.plot(range(len(values)), values, label=r"$L_f$", color="blue")
    plt.legend(loc="upper right")
    plt.ylabel('loss')

    ax2 = plt.subplot(312, sharex=ax1)
    values = np.array(losses["L_r"])
    plt.plot(range(len(values)), values, label=r"$L_r$", color="green")
    plt.legend(loc="upper right")
    plt.ylabel('loss')

    ax3 = plt.subplot(313, sharex=ax1)
    values = np.array(losses["L_f - L_r"])
    plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")
    plt.legend(loc="upper right")
    plt.xlabel('epochs')
    plt.ylabel('loss')


    path_png = os.path.join(save_path, 'losses-l={}.png'.format(lam))
    path_pdf = os.path.join(save_path, 'losses-l={}.pdf'.format(lam))

    plt.savefig(path_png, bbox_inches='tight')
    plt.savefig(path_pdf, bbox_inches='tight')

def main(args, config):
    # Set seed and import packages
    # NOTE: This need to be done before any keras module is imported!
    logger.debug("Import packages and set random seed to %s.",
                 int(config["seed"]))

    np.random.seed(int(config["seed"]))

    # Extract list of variables
    variables = config["variables"]
    logger.debug("Use variables:")
    for v in variables:
        logger.debug("%s", v)

    # Make sure output path exists
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load training dataset
    filename = config["datasets"][args.fold]
    logger.debug("Load training dataset from %s.", filename)
    with open(filename, "rb") as fd:
        X_train, X_test, y_train, y_test, z_train, z_test, scaler = pickle.load(fd)

    classes = config["classes"]

    # Add callbacks
    callbacks = []

    # Train model
    logger.info("Train keras model %s.", config["model"]["name"])

    if config["model"]["batch_size"] < 0:
        batch_size = X_train.shape[0]
    else:
        batch_size = config["model"]["batch_size"]

    lambda_parameter = config["model"]["lambda"]
    
    logger.info("Lambda parameter is {}".format(lambda_parameter))

    model_impl = getattr(pivot_models, config["model"]["name"])
    model = model_impl(len(variables), 1, z_train.shape[1], lambda_parameter)
    model.summary()
    model.pretrain_classifier(X_train, y_train, epochs=10, verbose=1)
    if lambda_parameter > 0.0:
        model.pretrain_adversary(X_train[y_train == 0], z_train[y_train == 0], epochs=10, verbose=1)

    history = model.fit(
        X_train,
        X_train[y_train == 0],
        y_train,
        z_train,
        z_train[y_train == 0],
        validation_data=(X_test, y_test, z_test),
        batch_size=batch_size,
        n_iter=config['model']['n_iter'])

    # Plot metrics

    plot_losses(history, save_path=output_path, lam = lambda_parameter)

    # Save model

    logger.info("Write model to {}.".format(output_path))
    model.save(output_path, args.fold)


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
