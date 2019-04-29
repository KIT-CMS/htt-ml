#!/usr/bin/env python

import logging
logger = logging.getLogger("keras_training")

import argparse
import yaml
import os
import pickle
from keras_custom_metrics import *

def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(
        description="Train machine Keras models for Htt analyses")
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

def draw_plots(variable_name, history, y_label, number_of_inputs):

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
                             "fold{}_{}_{}variables".format(args.fold, variable_name, number_of_inputs))
    plt.legend()
    plt.savefig(path_plot+".png", bbox_inches="tight")
    plt.savefig(path_plot+".pdf", bbox_inches="tight")


def draw_validation_losses(variable_names, history, y_label, number_of_inputs):
    # NOTE: Matplotlib needs to be imported after Keras/TensorFlow because of conflicting libraries
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.clf()
    for variable_name in variable_names:
        epochs = range(1, len(history.history[variable_name]) + 1)
        plt.plot(
            epochs, history.history["val_{}".format(variable_name)], lw=3, label="Validation {}".format(variable_name))
    plt.xlabel("Epoch"), plt.ylabel(y_label)
    path_plot = os.path.join(config["output_path"],
                             "fold{}_{}_{}variables".format(args.fold, 'validation_loss', number_of_inputs))
    plt.legend()
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")

def draw_training_losses(variable_names, history, y_label, number_of_inputs):
    # NOTE: Matplotlib needs to be imported after Keras/TensorFlow because of conflicting libraries
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.clf()
    for variable_name in variable_names:
        epochs = range(1, len(history.history[variable_name]) + 1)
        plt.plot(
            epochs, history.history["{}".format(variable_name)], lw=3, label="Train {}".format(variable_name))
    plt.xlabel("Epoch"), plt.ylabel(y_label)
    path_plot = os.path.join(config["output_path"],
                             "fold{}_{}_{}variables".format(args.fold, 'train_loss', number_of_inputs))
    plt.legend()
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")

def draw_custom_callbacks(metric_data, metric_names, variables, y_labels, number_of_inputs):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(metric_data) + 1)

    list_of_metrics = dict()
    for metric_name in metric_names:
        list_of_metrics[metric_name] = dict()
        for variable_name in variables:
            list_of_metrics[metric_name][variable_name] = []

    for data_point in metric_data:
        for metric_name in metric_names:
            for i_variable, variable_name in enumerate(variables):
                list_of_metrics[metric_name][variable_name].append(data_point[metric_name][i_variable])

    for metric_name, label_name in zip(metric_names, y_labels):
        plt.clf()
        for variable_name in variables:
            plt.plot(epochs, list_of_metrics[metric_name][variable_name], lw=3, label="{}".format(variable_name))
        plt.xlabel("Epoch"), plt.ylabel(label_name)
        plt.title("Validation score for {}".format(label_name))
        path_plot = os.path.join(config["output_path"],
                                 "fold{}_{}_{}variables".format(args.fold, metric_name, number_of_inputs))
        plt.legend()
        plt.savefig(path_plot+".png", bbox_inches="tight")
        plt.savefig(path_plot+".pdf", bbox_inches="tight")


def main(args, config):
    # Set seed and import packages
    # NOTE: This need to be done before any keras module is imported!
    logger.debug("Import packages and set random seed to %s.",
                 int(config["seed"]))
    import numpy as np
    np.random.seed(int(config["seed"]))

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
    import root_numpy

    from sklearn import preprocessing, model_selection
    import keras_models
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

    # Extract list of variables
    variables = config["variables"]
    logger.debug("Use variables:")
    for v in variables:
        logger.debug("%s", v)

    # Make sure output path exists
    output_path = config['output_path']
    if not os.path.exists(output_path):
        #os.mkdir(output_path)
        os.makedirs(output_path)

    output_path_json = config['output_path_json']
    if not os.path.exists(output_path_json):
        #os.mkdir(output_path_json)
        os.makedirs(output_path_json)


    # Load training dataset
    filename = config["datasets"][args.fold]
    logger.debug("Load training dataset from %s.", filename)
    x = []
    y = []
    w = []
    rfile = ROOT.TFile(filename, "READ")
    classes = config["classes"]
    for i_class, class_ in enumerate(classes):
        logger.debug("Process class %s.", class_)
        tree = rfile.Get(class_)
        if tree == None:
            logger.fatal("Tree %s not found in file %s.", class_, filename)
            raise Exception

        # Get inputs for this class
        #x_class = np.zeros((tree.GetEntries(), len(variables)))
        entries = tree.GetEntries() #if tree.GetEntries() < 8000 else 8000
        x_class = np.zeros((entries, len(variables)))
        x_conv = root_numpy.tree2array(tree, branches=variables)
        for i_var, var in enumerate(variables):
            x_class[:, i_var] = x_conv[var][:entries]
        x.append(x_class)

        # Get weights
        #w_class = np.zeros((tree.GetEntries(), 1))
        w_class = np.zeros((entries, 1))
        w_conv = root_numpy.tree2array(
            tree, branches=[config["event_weights"]])
        w_class[:,
                0] = w_conv[config["event_weights"]][:entries] * config["class_weights"][class_]
        w.append(w_class)

        # Get targets for this class
        #y_class = np.zeros((tree.GetEntries(), len(classes)))
        y_class = np.zeros((entries, len(classes)))
        y_class[:, i_class] = np.ones((entries))
        y.append(y_class)

    # Stack inputs, targets and weights to a Keras-readable dataset
    x = np.vstack(x)  # inputs
    y = np.vstack(y)  # targets
    w = np.vstack(w) * config["global_weight_scale"]  # weights
    w = np.squeeze(w)  # needed to get weights into keras

    # Perform input variable transformation and pickle scaler object
    logger.info("Use preprocessing method %s.", config["preprocessing"])
    if "standard_scaler" in config["preprocessing"]:
        scaler = preprocessing.StandardScaler().fit(x)
        for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
            logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                         var, mean, std)
    elif "no_scaler" in config["preprocessing"]:
        scaler = preprocessing.StandardScaler(with_mean=False, with_std=False).fit(x)

    elif "identity" in config["preprocessing"]:
        scaler = preprocessing.StandardScaler().fit(x)
        for i in range(len(scaler.mean_)):
            scaler.mean_[i] = 0.0
            scaler.scale_[i] = 1.0
        for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
            logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                         var, mean, std)
    elif "robust_scaler" in config["preprocessing"]:
        scaler = preprocessing.RobustScaler().fit(x)
        for var, mean, std in zip(variables, scaler.center_, scaler.scale_):
            logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                         var, mean, std)
    elif "min_max_scaler" in config["preprocessing"]:
        scaler = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0)).fit(x)
        for var, min_, max_ in zip(variables, scaler.data_min_,
                                   scaler.data_max_):
            logger.debug("Preprocessing (variable, min, max): %s, %s, %s", var,
                         min_, max_)
    elif "quantile_transformer" in config["preprocessing"]:
        scaler = preprocessing.QuantileTransformer(
            output_distribution="normal",
            random_state=int(config["seed"])).fit(x)
    else:
        logger.fatal("Preprocessing %s is not implemented.",
                     config["preprocessing"])
        raise Exception
    x = scaler.transform(x)

    path_preprocessing = os.path.join(
        config["output_path"],
        "fold{}_keras_preprocessing.pickle".format(args.fold))
    logger.info("Write preprocessing object to %s.", path_preprocessing)
    pickle.dump(scaler, open(path_preprocessing, 'wb'))

    # Split data in training and testing
    x_train, x_test, y_train, y_test, w_train, w_test = model_selection.train_test_split(
        x,
        y,
        w,
        test_size=1.0 - config["train_test_split"],
        random_state=int(config["seed"]))

    # Add callbacks
    callbacks = []
    if "early_stopping" in config["model"]:
        logger.info("Stop early after %s tries.",
                    config["model"]["early_stopping"])
        callbacks.append(
            EarlyStopping(patience=config["model"]["early_stopping"]))

    path_model = os.path.join(config["output_path"],
                              "fold{}_keras_model.h5".format(args.fold))
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
    metrics = significance(x_train=x_train, y_train=y_train, w_train=w_train)
    callbacks.append(metrics)

    # Train model
    if not hasattr(keras_models, config["model"]["name"]):
        logger.fatal("Model %s is not implemented.", config["model"]["name"])
        raise Exception
    logger.info("Train keras model %s.", config["model"]["name"])

    if config["model"]["batch_size"] < 0:
        batch_size = x_train.shape[0]
    else:
        batch_size = config["model"]["batch_size"]

    if config['model']['learning_rate'] < 1:
        learning_rate = config['model']['learning_rate']
    else:
        learning_rate = 1e-3

    # Adding up a list of y_train due to multiple losses
    y_train_list = []
    y_test_list = []
    for class_ in classes:
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    model_impl = getattr(keras_models, config["model"]["name"])
    model = model_impl(len(variables), len(classes), classes, learning_rate)
    model.summary()
    history = model.fit(
        [x_train, w_train],
        y_train_list,
        validation_data=([x_test, w_test], y_test_list),
        batch_size=batch_size,
        nb_epoch=config["model"]["epochs"],
        shuffle=True,
        callbacks=callbacks)

    # Plot metrics

    variable_names = []
    for class_ in classes:
        variable_names.append('{}_loss'.format(class_))

    draw_validation_losses(variable_names,history, y_label='Loss', number_of_inputs=len(variables))
    draw_training_losses(variable_names, history, y_label='Loss', number_of_inputs=len(variables))

    draw_plots(variable_name="loss", history=history, y_label="Loss", number_of_inputs=len(variables))


    if not "save_best_only" in config["model"]:
        logger.info("Write model to %s.", path_model)
        model.save(path_model)


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
