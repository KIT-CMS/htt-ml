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

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.models import load_model
import tensorflow as tf

from tensorflow_derivative.inputs import Inputs
from tensorflow_derivative.outputs import Outputs
from tensorflow_derivative.derivatives import Derivatives

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
    return yaml.load(open(filename, "r"))


def main(args, config_test, config_train):
    # Load preprocessing
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.info("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    # Load Keras model
    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.info("Load keras model %s.", path)
    model_keras = load_model(path)

    # Get TensorFlow graph
    inputs = Inputs(config_train["variables"])

    try:
        sys.path.append("htt-ml/training")
        import keras_models
    except:
        logger.fatal("Failed to import Keras models.")
        raise Exception
    try:
        name_keras_model = config_train["model"]["name"]
        model_tensorflow_impl = getattr(
            keras_models, config_train["model"]["name"] + "_tensorflow")
    except:
        logger.fatal(
            "Failed to load TensorFlow version of Keras model {}.".format(
                name_keras_model))
        raise Exception

    model_tensorflow = model_tensorflow_impl(inputs.placeholders, model_keras)
    outputs = Outputs(model_tensorflow, config_train["classes"])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Get operations for first-order derivatives
    deriv_ops = {}
    derivatives = Derivatives(inputs, outputs)
    for class_ in config_train["classes"]:
        deriv_ops[class_] = []
        for variable in config_train["variables"]:
            deriv_ops[class_].append(derivatives.get(class_, [variable]))

    # Loop over testing dataset
    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.info("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)
    mean_abs_deriv = {}
    for i_class, class_ in enumerate(config_train["classes"]):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception

        values = []
        for variable in config_train["variables"]:
            typename = tree.GetLeaf(variable).GetTypeName()
            if  typename == "Float_t":
                values.append(array("f", [-999]))
            elif typename == "Int_t":
                values.append(array("i", [-999]))
            else:
                logger.fatal("Variable {} has unknown type {}.".format(variable, typename))
                raise Exception
            tree.SetBranchAddress(variable, values[-1])

        if tree.GetLeaf(variable).GetTypeName() != "Float_t":
            logger.fatal("Weight branch has unkown type.")
            raise Exception
        weight = array("f", [-999])
        tree.SetBranchAddress(config_test["weight_branch"], weight)

        deriv_class = np.zeros((tree.GetEntries(),
                                len(config_train["variables"])))
        weights = np.zeros((tree.GetEntries()))

        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)

            # Preprocessing
            values_stacked = np.hstack(values).reshape(1, len(values))
            values_preprocessed = preprocessing.transform(values_stacked)

            # Keras inference
            response = model_keras.predict(values_preprocessed)
            response_keras = np.squeeze(response)

            # Tensorflow inference
            response = sess.run(
                model_tensorflow,
                feed_dict={
                    inputs.placeholders: values_preprocessed
                })
            response_tensorflow = np.squeeze(response)

            # Check compatibility
            mean_error = np.mean(np.abs(response_keras - response_tensorflow))

            if mean_error > 1e-5:
                logger.fatal(
                    "Found mean error of {} between Keras and TensorFlow output for event {}.".
                    format(mean_error, i_event))
                raise Exception

            # Calculate first-order derivatives
            deriv_values = sess.run(
                deriv_ops[class_],
                feed_dict={
                    inputs.placeholders: values_preprocessed
                })
            deriv_values = np.squeeze(deriv_values)
            deriv_class[i_event, :] = deriv_values

            # Store weight
            weights[i_event] = weight[0]

        if args.no_abs:
            mean_abs_deriv[class_] = np.average((deriv_class), weights=weights, axis=0)
        else:
            mean_abs_deriv[class_] = np.average(np.abs(deriv_class), weights=weights, axis=0)

    # Normalize rows
    classes = config_train["classes"]
    matrix = np.vstack([mean_abs_deriv[class_] for class_ in classes])
    if args.normalize:
        for i_class, class_ in enumerate(classes):
            matrix[i_class, :] = matrix[i_class, :] / np.sum(
                matrix[i_class, :])

    # Plotting
    variables = config_train["variables"]
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
    plt.xlim(0, len(config_train["variables"]))
    plt.ylim(0, len(config_train["classes"]))
    output_path = os.path.join(config_train["output_path"],
                               "fold{}_keras_taylor_1D".format(args.fold))
    logger.info("Save plot to {}.".format(output_path))
    plt.savefig(output_path+".png", bbox_inches='tight')
    plt.savefig(output_path+".pdf", bbox_inches='tight')


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
