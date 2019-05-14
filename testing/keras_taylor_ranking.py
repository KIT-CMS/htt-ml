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

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 20
import matplotlib.pyplot as plt

sys.path.append('htt-ml/utils')

import tensorflow as tf

from keras_to_tensorflow import get_tensorflow_model

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
    parser.add_argument(
        "--no-abs",
        action="store_true",
        default=False,
        help="Do not use abs for metric.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))

class derivative_operation(object):
    def __init__(self, outputs, input, class_name):
        self.derivative = outputs.outputs_dict[class_name]

        self.first_order_gradients = tf.gradients(self.derivative, input)[0][0]
        self.first_order_gradients = tf.unstack(self.first_order_gradients)
        self.second_order_gradient = [tf.gradients(first_order_gradient, input) for first_order_gradient in self.first_order_gradients]



def main(args, config_test, config_train):
    # Load preprocessing
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.info("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    classes = config_train["classes"]
    variables = config_train["variables"]

    model_keras, tensorflow_model, outputs, tf_input, tf_output, dropout_name = get_tensorflow_model(args,
                                                                                                     config_train,
                                                                                                     config_test)
    if dropout_name:
        dropout = tensorflow_model.get_tensor_by_name(dropout_name)
    input = tensorflow_model.get_tensor_by_name(tf_input)
    output = tensorflow_model.get_tensor_by_name(tf_output)

    sess = tf.Session(graph=tensorflow_model)

    #Get names for first-order and second-order derivatives
    logger.debug("Set up derivative names.")
    deriv_ops_names = []
    for variable in variables:
        deriv_ops_names.append([variable])
    for i, i_var in enumerate(variables):
        for j, j_var in enumerate(variables):
            if j < i:
                continue
            deriv_ops_names.append([i_var, j_var])

    # Loop over testing dataset
    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.info("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)
    deriv_class = {}
    weights = {}
    deriv_ops = {}

    for i_class, class_ in enumerate(classes):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception

        deriv_ops[class_] = derivative_operation(outputs=outputs, input=input, class_name=class_)

    time_start = time.time()

    for i_class, class_ in enumerate(classes):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception
        friend_trees_names = [k.GetName() for k in file_.GetListOfKeys() if "_".join([class_,"friend"]) in k.GetName()]
        for friend in friend_trees_names:
            tree.AddFriend(friend)

        values = []
        for variable in variables:
            typename = tree.GetLeaf(variable).GetTypeName()
            if typename == "Float_t":
                values.append(array("f", [-999]))
            elif typename == "Int_t":
                values.append(array("i", [-999]))
            else:
                logger.fatal("Variable {} has unknown type {}.".format(
                    variable, typename))
                raise Exception
            tree.SetBranchAddress(variable, values[-1])

        if tree.GetLeaf(variable).GetTypeName() != "Float_t":
            logger.fatal("Weight branch has unkown type.")
            raise Exception
        weight = array("f", [-999])
        tree.SetBranchAddress(config_test["weight_branch"], weight)

        length_deriv_class = (len(variables)**2 + len(variables))/2 + len(variables)

        deriv_class[class_] = np.zeros((tree.GetEntries(),
                                       length_deriv_class))
        weights[class_] = np.zeros((tree.GetEntries()))

        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)

            # Preprocessing
            values_stacked = np.hstack(values).reshape(1, len(values))
            values_preprocessed = preprocessing.transform(values_stacked)

            # Keras inference
            response = model_keras.predict(values_preprocessed)
            response_keras = np.squeeze(response)

            deriv_op = deriv_ops[class_]

            if dropout_name:
                feed_dict = {
                    input: values_preprocessed,
                    dropout: False
                }
            else:
                feed_dict = {
                    input: values_preprocessed
                }

            # Tensorflow inference
            response = sess.run(
                output,
                feed_dict=feed_dict)
            response_tensorflow = np.squeeze(response)

            # Check compatibility
            mean_error = np.mean(np.abs(response_keras - response_tensorflow))

            if mean_error > 1e-5:
                logger.fatal(
                    "Found mean error of {} between Keras and TensorFlow output for event {}.".
                        format(mean_error, i_event))
                raise Exception

            first_order_values = sess.run(deriv_op.first_order_gradients, feed_dict=feed_dict)

            second_order_values = sess.run(deriv_op.second_order_gradient, feed_dict=feed_dict)

            second_order_stacked = np.stack(np.squeeze(second_order_values))
            lower_hessian_half = second_order_stacked[np.triu_indices(len(variables))]
            deriv_values = np.concatenate((np.squeeze(first_order_values), lower_hessian_half))

            deriv_values = np.squeeze(deriv_values)
            deriv_class[class_][i_event, :] = deriv_values

            # Store weight
            weights[class_][i_event] = weight[0]

            if i_event % 10000 == 0:
                time_between = time.time()
                logger.info('Processing event {}'.format(i_event))
                logger.info('Current time: {}'.format(time_between-time_start))

    time_end = time.time()
    logger.info('Elapsed time: {}'.format((time_end - time_start)/60.))

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

        yx = zip(ranking_tmp, labels_tmp)
        yx.sort(reverse=True)
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

        yx = zip(ranking_tmp, labels_tmp)
        yx.sort(reverse=True)
        labels_tmp = [x for y, x in yx]
        ranking_tmp = [y for y, x in yx]

        ranking_singles[class_] = ranking_tmp
        labels_singles[class_] = labels_tmp

    # Write table
    for class_ in classes + ["all"]:
        output_path = os.path.join(config_train["output_path"],
                                   "fold{}_keras_taylor_ranking_{}.txt".format(
                                       args.fold, class_))
        logger.info("Save table to {}.".format(output_path))
        f = open(output_path, "w")
        for rank, (label, score) in enumerate(
                zip(labels[class_], ranking[class_])):
            f.write("{0:<4} : {1:<60} : {2:g}\n".format(rank, label, score))

    # Write table
    for class_ in classes + ["all"]:
        output_path = os.path.join(config_train["output_path"],
                                   "fold{}_keras_taylor_1D_{}.txt".format(
                                       args.fold, class_))
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
    output_path = os.path.join(config_train["output_path"],
                               "fold{}_keras_taylor_ranking.yaml".format(
                                   args.fold))
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
        output_path = os.path.join(config_train["output_path"],
                                   "fold{}_keras_taylor_ranking_{}.png".format(
                                       args.fold, class_))
        logger.info("Save plot to {}.".format(output_path))
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
