#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
from array import array
import yaml
import pickle
import numpy as np
import os
import time
import sys
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 20
import matplotlib.pyplot as plt

sys.path.append('htt-ml/utils')

import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from pandas.api.types import CategoricalDtype 


import logging
logger = logging.getLogger("keras_taylor_ranking")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate taylor coefficients.")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_testing", help="Path to testing config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    parser.add_argument("--era", required=False, type=int, default=None, help="Era to be tested.")
    parser.add_argument(
        "--no-abs",
        action="store_true",
        default=False,
        help="Do not use abs for metric.")
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

    classes = config_train["classes"]
    variables = config_train["variables"]

    # Define eras
    if args.era:
        eras = ["2016", "2017", "2018"]
        path = os.path.join(config_train["datasets_{}".format(args.era)][(1, 0)[args.fold]])
    else:
        eras = []
        path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    
    path_model = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.info("Load keras model %s.", path_model)
    model_keras = load_model(path_model)

    #Get names for first-order and second-order derivatives
    logger.debug("Set up derivative names.")
    deriv_ops_names = []
    for variable in variables + eras:
        deriv_ops_names.append([variable])
    for i, i_var in enumerate(variables + eras):
        for j, j_var in enumerate(variables + eras):
            if j < i:
                continue
            deriv_ops_names.append([i_var, j_var])

    # Function to compute gradients of model answers in optimized graph mode    
    @tf.function
    def get_gradients(model, samples, output_ind):
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

    # Function to compute hessian of model answers in optimized graph mode   
    # @tf.function
    def get_hessians(model, sample, output_ind):
        # Define function to get single hessian
        def get_single_hessian(single_sample):
            single_sample = tf.expand_dims(single_sample, axis=0)
            #Function to compute gradient of a vector in regard to inputs (on outer tape)
            def tot_gradient(vector):
                return tape.gradient(vector, single_sample)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(single_sample)
                with tf.GradientTape(persistent=True) as tape_of_tape:
                    tape_of_tape.watch(single_sample)
                    # Get response from model with (only choosen output class)
                    response = model(single_sample, training=False)[:, output_ind]
                # Get gradient of choosen output class wrt. input sample
                grads = tape_of_tape.gradient(response, single_sample)
                # Compute hessian of model from gradients of model wrt. input sample
                hessian = tf.map_fn(tot_gradient, grads[0])
            return hessian
        # Apply function to every sample to get all hessians
        hessians = tf.vectorized_map(get_single_hessian, sample_tensor)
        return hessians

    # Loop over testing dataset
    logger.info("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)
    deriv_class = {}
    weights = {}

    for i_class, class_ in enumerate(classes):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception


    for i_class, class_ in enumerate(classes):
        logger.debug("Current time: {}".format(datetime.now().strftime("%H:%M:%S")))
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception
        friend_trees_names = [k.GetName() for k in file_.GetListOfKeys() if k.GetName().startswith("_".join([class_,"friend"]))]
        for friend in friend_trees_names:
            tree.AddFriend(friend)

        for variable in variables:
            typename = tree.GetLeaf(variable).GetTypeName()
            if not (typename == "Float_t" or typename == "Int_t"):
                logger.fatal("Variable {} has unknown type {}.".format(variable, typename))
                raise Exception
        if tree.GetLeaf(variable).GetTypeName() != "Float_t":
            logger.fatal(
                "Weight branch has unkown type: {}".format(tree.GetLeaf(config_test["weight_branch"]).GetTypeName()))
            raise Exception

        length_variables = len(variables) + len(eras)
        length_deriv_class = (length_variables**2 + length_variables)/2 + length_variables

        # Function to get upper triangle of matrix for every matrix in array
        def triu_map(matrix_array, size):
            # get upper triangle indices for given matrix size
            triu = np.triu_indices(size)
            # Function to apply indices to single element of matrix array and get single output 
            def single_triu(single_mat):
                return single_mat[triu]
            # Apply function to every element
            return np.array(list(map(single_triu, matrix_array)))

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
        # Transform numpy array with samples to tensorflow tensor
        sample_tensor = tf.convert_to_tensor(values_preprocessed)
        # Get array of gradients of model wrt. samples 
        gradients = tf.squeeze(get_gradients(model_keras, sample_tensor, i_class))
        # Get array of hessians of model wrt. samples 
        hessians = tf.squeeze(get_hessians(model_keras, sample_tensor, i_class))
        # Get array of upper triangles of hessians of model wrt. samples 
        upper_hessian_half = triu_map(hessians.numpy(), length_variables)
        # Append gradient values to hessian values
        deriv_values = np.concatenate((gradients, upper_hessian_half), axis=1)
        # Collect results for class
        deriv_class[class_] = deriv_values
        weights[class_] = flat_weight
        
    # Calculate taylor coefficients
    mean_abs_deriv = {}
    for class_ in classes:
        if args.no_abs:
            mean_abs_deriv[class_] = np.average(
                (deriv_class[class_]), weights=weights[class_], axis=0)
        else:
            mean_abs_deriv[class_] = np.average(
                np.abs(deriv_class[class_]), weights=weights[class_], axis=0)

    deriv_all = np.vstack([deriv_class[class_] for class_ in classes])
    weights_all = np.hstack([weights[class_] for class_ in classes])
    if args.no_abs:
        mean_abs_deriv_all = np.average(
            (deriv_all), weights=weights_all, axis=0)
    else:
        mean_abs_deriv_all = np.average(
            np.abs(deriv_all), weights=weights_all, axis=0)
    mean_abs_deriv["all"] = mean_abs_deriv_all

    # Get ranking
    ranking = {}
    labels = {}
    for class_ in classes + ["all"]:
        labels_tmp = []
        ranking_tmp = []
        for names, value in zip(deriv_ops_names, mean_abs_deriv[class_]):
            labels_tmp.append(", ".join(names))
            if len(names) == 2:
                if names[0] == names[1]:
                    ranking_tmp.append(0.5 * value)
                else:
                    ranking_tmp.append(value)
            else:
                ranking_tmp.append(value)

        yx = list(zip(ranking_tmp, labels_tmp))
        yx = sorted(yx, reverse=True)
        labels_tmp = [x for y, x in yx]
        ranking_tmp = [y for y, x in yx]

        ranking[class_] = ranking_tmp
        labels[class_] = labels_tmp

    ranking_singles = {}
    labels_singles = {}
    for class_ in classes + ["all"]:
        labels_tmp = []
        ranking_tmp = []
        for names, value in zip(deriv_ops_names, mean_abs_deriv[class_]):
            if len(names) > 1:
                continue
            labels_tmp.append(", ".join(names))
            ranking_tmp.append(value)

        yx = list(zip(ranking_tmp, labels_tmp))
        yx = sorted(yx, reverse=True)
        labels_tmp = [x for y, x in yx]
        ranking_tmp = [y for y, x in yx]

        ranking_singles[class_] = ranking_tmp
        labels_singles[class_] = labels_tmp

    # Write table
    for class_ in classes + ["all"]:
        plot_name_era = "_{}".format(args.era) if args.era else ""
        output_path = os.path.join(config_train["output_path"],
                                   "fold{}_keras_taylor_ranking_{}{}.txt".format(
                                       args.fold, class_, plot_name_era))
        logger.info("Save table to {}.".format(output_path))
        f = open(output_path, "w")
        for rank, (label, score) in enumerate(
                zip(labels[class_], ranking[class_])):
            f.write("{0:<4} : {1:<60} : {2:g}\n".format(rank, label, score))

    # Write table
    for class_ in classes + ["all"]:
        plot_name_era = "_{}".format(args.era) if args.era else ""
        output_path = os.path.join(config_train["output_path"],
                                   "fold{}_keras_taylor_1D_{}{}.txt".format(
                                       args.fold, class_, plot_name_era))
        logger.info("Save table to {}.".format(output_path))
        f = open(output_path, "w")
        for rank, (label, score) in enumerate(
                zip(labels_singles[class_], ranking_singles[class_])):
            f.write("{0:<4} : {1:<60} : {2:g}\n".format(rank, label, score))

    # Store results for combined metric in file
    output_yaml = []
    for names, score in zip(labels["all"], ranking["all"]):
        output_yaml.append({
            "variables": names.split(", "),
            "score": float(score)
        })
    plot_name_era = "_{}".format(args.era) if args.era else ""
    output_path = os.path.join(config_train["output_path"],
                               "fold{}_keras_taylor_ranking{}.yaml".format(
                                   args.fold, plot_name_era))
    yaml.dump(output_yaml, open(output_path, "w"), default_flow_style=False)
    logger.info("Save results to {}.".format(output_path))

    # Plotting
    for class_ in classes + ["all"]:
        plt.figure(figsize=(7, 4))
        ranks_1d = []
        ranks_2d = []
        scores_1d = []
        scores_2d = []
        for i, (label, score) in enumerate(
                zip(labels[class_], ranking[class_])):
            if ", " in label:
                scores_2d.append(score)
                ranks_2d.append(i)
            else:
                scores_1d.append(score)
                ranks_1d.append(i)
        plt.clf()

        plt.plot(
            ranks_2d,
            scores_2d,
            "+",
            mew=10,
            ms=3,
            label="Second-order features",
            alpha=1.0)
        plt.plot(
            ranks_1d,
            scores_1d,
            "+",
            mew=10,
            ms=3,
            label="First-order features",
            alpha=1.0)
        plt.xlabel("Rank")
        plt.ylabel("$\\langle t_{i} \\rangle$")
        plt.legend()
        plot_name_era = "_{}".format(args.era) if args.era else ""
        output_path = os.path.join(config_train["output_path"],
                                   "fold{}_keras_taylor_ranking_{}{}.png".format(
                                       args.fold, class_, plot_name_era))
        logger.info("Save plot to {}.".format(output_path))
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
