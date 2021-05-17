#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
from array import array
import yaml
import pickle
import numpy as np
import os
import sys
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from matplotlib import cm

from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
from pandas.api.types import CategoricalDtype 


import logging
logger = logging.getLogger("keras_taylor_1D")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Produce confusion matrice")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_testing", help="Path to testing config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    parser.add_argument("--era", required=False, type=int, default=None, help="Era to be tested.")
    parser.add_argument(
        "--no-abs",
        action="store_true",
        default=False,
        help="Do not use abs for metric.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize rows.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"), Loader=yaml.SafeLoader)


def main(args, config_test, config_train):
    logger.info("Tensorflow version: {}".format(tf.__version__))

    # Load preprocessing
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.info("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"), encoding="bytes")
    # Load Keras model
    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.info("Load keras model %s.", path)
    model_keras = load_model(path)

    # Define eras
    if args.era:
        eras = ["2016", "2017", "2018"]
        path = os.path.join(config_train["datasets_{}".format(args.era)][(1, 0)[args.fold]])
    else:
        eras = []
        path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])

    # Get TensorFlow graph

    try:
        sys.path.append("htt-ml/training")
        import keras_models
    except:
        logger.fatal("Failed to import Keras models.")
        raise Exception
    
    # Function to compute gradients of model answers in optimized graph mode    
    @tf.function(experimental_relax_shapes=True)
    def get_gradients(model, samples, output_ind):
        # Transform numpy array with samples to tensorflow tensor
        sample_tensor = tf.convert_to_tensor(samples)
        # Define function to get single gradient
        def get_single_gradient(single_sample):
            # (Expand shape of sample for gradient function)
            single_sample = tf.expand_dims(single_sample, axis=0)
            with tf.GradientTape() as tape:
                tape.watch(single_sample)
                # Get response from model with (only choosen output class)
                response = model(single_sample, training=False)[:, output_ind]
            # Get gradient of choosen output class wrt. input sample
            grad = tape.gradient(response, single_sample)
            return grad
        # Apply function to every sample to get all gradients
        grads = tf.vectorized_map(get_single_gradient, sample_tensor)
        return grads


    # Loop over testing dataset
    logger.info("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)
    mean_abs_deriv = {}
    for i_class, class_ in enumerate(config_train["classes"]):
        logger.debug("Current time: {}".format(datetime.now().strftime("%H:%M:%S")))
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception
        friend_trees_names = [k.GetName() for k in file_.GetListOfKeys() if k.GetName().startswith("_".join([class_,"friend"]))]
        for friend in friend_trees_names:
            tree.AddFriend(friend)

        variables = config_train["variables"]
        for variable in variables:
            typename = tree.GetLeaf(variable).GetTypeName()
            if not (typename == "Float_t" or typename == "Int_t"):
                logger.fatal("Variable {} has unknown type {}.".format(variable, typename))
                raise Exception
        if tree.GetLeaf(variable).GetTypeName() != "Float_t":
            logger.fatal(
                "Weight branch has unkown type: {}".format(tree.GetLeaf(config_test["weight_branch"]).GetTypeName()))
            raise Exception

        # Convert tree to numpy array and labels for variable columns and weight column
        tdata, tcolumns = tree.AsMatrix(return_labels=True ,columns=variables)
        wdata, wcolumns = tree.AsMatrix(return_labels=True ,columns=[config_test["weight_branch"]])
        # Convert numpy array and labels to pandas dataframe for variable columns
        pdata = pd.DataFrame(data=tdata, columns=tcolumns)
        # Flatten weight column into 1d array
        flat_weight = wdata.flatten()
        # Apply preprocessing of training to variables
        values_preprocessed = pd.DataFrame(data=preprocessing.transform(pdata), columns=tcolumns)
        # Check for viable era
        if args.era:
            if not args.era in [2016, 2017, 2018]:
                logger.fatal("Era must be 2016, 2017 or 2018 but is {}".format(args.era))
                raise Exception
            # Append one hot encoded era information to variables
            # Append as int
            values_preprocessed["era"] = args.era
            # Expand to possible values (necessary for next step)
            values_preprocessed["era"] = values_preprocessed["era"].astype(CategoricalDtype([2016, 2017, 2018]))
            # Convert to one hot encoding and append
            values_preprocessed = pd.concat([values_preprocessed, pd.get_dummies(values_preprocessed["era"], prefix="era")], axis=1)
            # Remove int era column
            values_preprocessed.drop(["era"], axis=1, inplace=True)
            ###
        # Get array of gradients of model wrt. samples 
        gradients = tf.squeeze(get_gradients(model_keras, values_preprocessed, i_class))
        # Get weighted average of gradients/ abs of gradients
        if args.no_abs:
            mean_abs_deriv[class_] = np.average(gradients, weights=flat_weight, axis=0)
        else:
            mean_abs_deriv[class_] = np.average(np.abs(gradients), weights=flat_weight, axis=0)

    # Normalize rows
    classes = config_train["classes"]
    matrix = np.vstack([mean_abs_deriv[class_] for class_ in classes])
    if args.normalize:
        for i_class, class_ in enumerate(classes):
            matrix[i_class, :] = matrix[i_class, :] / np.sum(
                matrix[i_class, :])

    # Plotting
    variables = config_train["variables"] + eras
    plt.figure(0, figsize=(len(variables), len(classes)))
    axis = plt.gca()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j + 0.5,
                i + 0.5,
                '{:.2f}'.format(matrix[i, j]),
                ha='center',
                va='center')
    q = plt.pcolormesh(matrix, cmap='Wistia')
    #cbar = plt.colorbar(q)
    #cbar.set_label("mean(abs(Taylor coefficients))", rotation=270, labelpad=20)
    plt.xticks(
        np.array(range(len(variables))) + 0.5, variables, rotation='vertical')
    plt.yticks(
        np.array(range(len(classes))) + 0.5, classes, rotation='horizontal')
    plt.xlim(0, len(variables))
    plt.ylim(0, len(classes))
    plot_name_era = "_{}".format(args.era) if args.era else ""
    output_path = os.path.join(config_train["output_path"],
                               "fold{}_keras_taylor_1D{}".format(args.fold, plot_name_era))
    logger.info("Save plot to {}.".format(output_path))
    plt.savefig(output_path+".png", bbox_inches='tight')
    plt.savefig(output_path+".pdf", bbox_inches='tight')


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
