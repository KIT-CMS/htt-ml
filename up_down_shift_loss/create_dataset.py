#!/usr/bin/env python

import logging

logger = logging.getLogger("pivot_training")

import argparse
import yaml
import os
import pickle
import numpy as np


def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(
        description="Train pivot adversarial network")
    parser.add_argument("config_training", help="Path to training config file")
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

def main(args, config):
    # Set seed and import packages
    # NOTE: This need to be done before any keras module is imported!
    logger.debug("Import packages and set random seed to %s.",
                 int(config["seed"]))

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
    import root_numpy

    from sklearn import preprocessing, model_selection

    np.random.seed(int(config["seed"]))

    # Make sure output path exists
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    variables = config["variables"]

    # Load training dataset
    filename = config["datasets"][args.fold]
    logger.debug("Load training dataset from %s.", filename)

    x = []
    y = []
    w = []
    z_up = []
    z_down = []
    rfile = ROOT.TFile(filename, "READ")
    classes = config["classes"]
    for i_class, class_ in enumerate(classes):
        logger.debug("Process class %s.", class_)
        tree = rfile.Get(class_)
        if tree == None:
            logger.fatal("Tree %s not found in file %s.", class_, filename)
            raise Exception

        # Get inputs for this class
        x_class = np.zeros((tree.GetEntries(), len(variables)))
        x_conv = root_numpy.tree2array(tree, branches=variables)
        for i_var, var in enumerate(variables):
            x_class[:, i_var] = x_conv[var]
        x.append(x_class)

        # Get training weights
        w_class = np.zeros((tree.GetEntries(), 1))
        w_conv = root_numpy.tree2array(
            tree, branches=[config["event_weights"]])
        # if class_ == 'ggh':
        #     w_class[:,0] = w_conv[config["event_weights"]] * config["class_weights"][class_] / 3.
        #     w.append(w_class)
        #     w.append(w_class)
        # else:
        w_class[:, 0] = w_conv[config["event_weights"]] * config["class_weights"][class_]
        w.append(w_class)

        # Get systematic uncertainty

        z_class = np.ones((tree.GetEntries(),1))
        if class_ == 'ggh':
            logger.info('Getting up and down weights for class {}'.format(class_))
            uncertainty_class = np.zeros((tree.GetEntries()))
            uncertainty_conv = root_numpy.tree2array(
                tree, branches=[config['target_uncertainty']]
            )
            z_class_up = np.ones((tree.GetEntries(),1))
            z_class_up[:,0] = uncertainty_conv[config['target_uncertainty']]
            z_up.append(z_class_up)

            z_class_down = np.ones((tree.GetEntries(),1))
            z_class_down[:,0] = 1./uncertainty_conv[config['target_uncertainty']]
            z_down.append(z_class_down)
        else:
            z_up.append(z_class)
            z_down.append(z_class)

        # Get targets for this class
        y_class = np.zeros((tree.GetEntries(), len(classes)))
        y_class[:, i_class] = np.ones((tree.GetEntries()))
        y.append(y_class)

    # Stack inputs, targets and weights to a Keras-readable dataset
    x = np.vstack(x)  # inputs
    y = np.vstack(y)  # targets
    z_up = np.vstack(z_up).squeeze()  # adversary targets
    z_down = np.vstack(z_down).squeeze() # adversary targets
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
    x_train, x_test, y_train, y_test, w_train, w_test, z_up_train, z_up_test, z_down_train, z_down_test = model_selection.train_test_split(
        x,
        y,
        w,
        z_up,
        z_down,
        test_size=1.0 - config["train_test_split"],
        random_state=int(config["seed"]))

    path = os.path.join(config['output_path'], 'fold{}_train.pickle'.format(args.fold))
    pickle.dump([x_train, y_train, w_train, z_up_train, z_down_train], open(path, 'wb'))

    path = os.path.join(config['output_path'], 'fold{}_test.pickle'.format(args.fold))
    pickle.dump([x_test, y_test, w_test, z_up_test, z_down_test], open(path, 'wb'))

if __name__ == "__main__":
    args = parse_arguments()
    config_train = parse_config(args.config_training)
    main(args, config_train)