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
    z = []
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
        if class_ == 'ggh':
            w_class[:,0] = w_conv[config["event_weights"]] * config["class_weights"][class_] / 3.
            w.append(w_class)
            w.append(w_class)
        else:
            w_class[:, 0] = w_conv[config["event_weights"]] * config["class_weights"][class_]
        w.append(w_class)

        # Get systematic uncertainty

        z_class = np.zeros((tree.GetEntries(),2))
        z_class[:,0] = 1
        z.append(z_class)
        if class_ == 'ggh':
            logger.info('Applying uncertainties to input parameters for class {}'.format(class_))
            uncertainty_class = np.zeros((tree.GetEntries()))
            uncertainty_conv = root_numpy.tree2array(
                tree, branches=[config['target_uncertainty']]
            )
            uncertainty_class[:] = uncertainty_conv[config['target_uncertainty']]
            print(uncertainty_class)
            x_up = x_class
            for j,uncertainty in enumerate(uncertainty_class):
                x_up[j] *= uncertainty
            x.append(x_up)
            z_class_up = np.zeros((tree.GetEntries(),2))
            z_class_up[:,1] = 1
            z.append(z_class_up)
            x_down = x_class
            for j,uncertainty in enumerate(uncertainty_class):
                x_down[j] *= 1./float(uncertainty)
            x.append(x_down)
            z_class_down = np.zeros((tree.GetEntries(),2))
            z_class_down[:,1] = 1
            z.append(z_class_down)

        # Get targets for this class
        y_class = np.zeros((tree.GetEntries(), len(classes)))
        y_class[:, i_class] = np.ones((tree.GetEntries()))
        if class_ == 'ggh':
            y.append(y_class)
            y.append(y_class)
        y.append(y_class)

    # Stack inputs, targets and weights to a Keras-readable dataset
    x = np.vstack(x)  # inputs
    y = np.vstack(y)  # targets
    z = np.vstack(z)  # adversary targets
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
    x_train, x_test, y_train, y_test, w_train, w_test, z_train, z_test = model_selection.train_test_split(
        x,
        y,
        w,
        z,
        test_size=1.0 - config["train_test_split"],
        random_state=int(config["seed"]))

    # Train model
    logger.info("Train keras model %s.", config["model"]["name"])

    if config["model"]["batch_size"] < 0:
        batch_size = x_train.shape[0]
    else:
        batch_size = config["model"]["batch_size"]

    lambda_parameter = config["model"]["lambda"]

    logger.info("Lambda parameter is {}".format(lambda_parameter))

    epochs = config['model']['epochs']
    dropout = config['model']['dropout']

    model_impl = getattr(pivot_models, config["model"]["name"])

    num_adv_outputs = z_train.shape[1]

    model = model_impl(x_train.shape[1], len(classes), num_adv_outputs, lambda_parameter, dropout)
    model.summary()
    model.pretrain_classifier(x_train, y_train, sample_weights=w_train, batch_size=batch_size, path=output_path, fold=args.fold, epochs=epochs, verbose=1)
    if lambda_parameter > 0.0:
        model.pretrain_adversary(x_train[(y_train[:,0] == 1)], z_train[(y_train[:,0] == 1)], class_weights={0: 2., 1: 1.}, epochs=epochs, verbose=1)

    w_train_adv = np.ones((len(w_train)))
    w_train_adv[(z_train[:, 0] == 1)] = 2.

    history = model.fit(
        x_train,
        x_train[(y_train[:,0] == 1)],
        y_train,
        z_train,
        z_train[(y_train[:,0] == 1)],
        sample_weights=[w_train, w_train_adv],
        class_weights_adv = {0: 2.,1: 1.},
        validation_data=(x_test, y_test, z_test, w_test),
        batch_size=batch_size,
        n_iter=config['model']['n_iter'])

    # Plot metrics

    plot_losses(history, save_path=output_path, lam=lambda_parameter)

    # Save model

    logger.info("Write model to {}.".format(output_path))
    model.save(output_path, args.fold)


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
