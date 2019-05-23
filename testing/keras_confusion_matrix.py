#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
from array import array
import yaml
import pickle
import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.models import load_model

import logging
logger = logging.getLogger("keras_confusion_matrix")
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
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))


def get_efficiency_representations(m):
    ma = np.zeros(m.shape)
    mb = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ma[i, j] = m[i, j] / m[i, i]
            mb[i, j] = m[i, j] / np.sum(m[i, :])
    return ma, mb


def get_purity_representations(m):
    ma = np.zeros(m.shape)
    mb = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ma[i, j] = m[i, j] / m[j, j]
            mb[i, j] = m[i, j] / np.sum(m[:, j])
    return ma, mb


def plot_confusion(confusion, classes, filename, label, markup='{:.2f}'):
    logger.debug("Write plot to %s.", filename)
    plt.figure(figsize=(2.5 * confusion.shape[0], 2.0 * confusion.shape[1]))
    axis = plt.gca()
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axis.text(
                i + 0.5,
                j + 0.5,
                markup.format(confusion[i, -1 - j]),
                ha='center',
                va='center')
    q = plt.pcolormesh(np.transpose(confusion)[::-1], cmap='Wistia')
    cbar = plt.colorbar(q)
    cbar.set_label(label, rotation=270, labelpad=50)
    plt.xticks(
        np.array(range(len(classes))) + 0.5, classes, rotation='vertical')
    plt.yticks(
        np.array(range(len(classes))) + 0.5,
        classes[::-1],
        rotation='horizontal')
    plt.xlim(0, len(classes))
    plt.ylim(0, len(classes))
    plt.ylabel('Predicted class')
    plt.xlabel('True class')
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.savefig(filename+".pdf", bbox_inches='tight')

    d = {}
    for i1, c1 in enumerate(classes):
        d[c1] = {}
        for i2, c2 in enumerate(classes):
            d[c1][c2] = float(confusion[i1, i2])
    f = open(filename+".yaml", "w")
    yaml.dump(d, f)


def print_matrix(p, title):
    stdout.write(title + '\n')
    for i in range(p.shape[0]):
        stdout.write('    ')
        for j in range(p.shape[1]):
            stdout.write('{:.4f} & '.format(p[i, j]))
        stdout.write('\b\b\\\\\n')


def main(args, config_test, config_train):
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.info("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.info("Load keras model %s.", path)
    model = load_model(path)

    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.info("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)
    confusion = np.zeros(
        (len(config_train["classes"]), len(config_train["classes"])),
        dtype=np.float)
    confusion2 = np.zeros(
        (len(config_train["classes"]), len(config_train["classes"])),
        dtype=np.float)
    class_weights = config_train["class_weights"]
    for i_class, class_ in enumerate(config_train["classes"]):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception
        friend_trees_names = [k.GetName() for k in file_.GetListOfKeys() if "_".join([class_,"friend"]) in k.GetName()]
        for friend in friend_trees_names:
            tree.AddFriend(friend)

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

        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)
            values_stacked = np.hstack(values).reshape(1, len(values))
            values_preprocessed = preprocessing.transform(values_stacked)
            response = model.predict(values_preprocessed)
            response = np.squeeze(response)
            max_index = np.argmax(response)
            confusion[i_class, max_index] += weight[0]
            confusion2[i_class, max_index] += weight[0]*class_weights[class_]

    # Debug output to ensure that plotting is correct
    for i_class, class_ in enumerate(config_train["classes"]):
        logger.debug("True class: {}".format(class_))
        for j_class, class2 in enumerate(config_train["classes"]):
            logger.debug("Predicted {}: {}".format(class2, confusion[i_class, j_class]))

    # Plot confusion matrix
    logger.info("Write confusion matrices.")
    path_template = os.path.join(config_train["output_path"],
                                 "fold{}_keras_confusion_{}")

    plot_confusion(confusion, config_train["classes"],
                   path_template.format(args.fold, "standard"), "Arbitrary unit")
    plot_confusion(confusion2, config_train["classes"],
                   path_template.format(args.fold, "standard_cw"), "Arbitrary unit")

    confusion_eff1, confusion_eff2 = get_efficiency_representations(confusion)
    confusion_eff3, confusion_eff4 = get_efficiency_representations(confusion2)
    plot_confusion(confusion_eff1, config_train["classes"],
                   path_template.format(args.fold, "efficiency1"), "Efficiency")
    plot_confusion(confusion_eff2, config_train["classes"],
                   path_template.format(args.fold, "efficiency2"), "Efficiency")
    plot_confusion(confusion_eff3, config_train["classes"],
                   path_template.format(args.fold, "efficiency_cw1"), "Efficiency")
    plot_confusion(confusion_eff4, config_train["classes"],
                   path_template.format(args.fold, "efficiency_cw2"), "Efficiency")

    confusion_pur1, confusion_pur2 = get_purity_representations(confusion)
    confusion_pur3, confusion_pur4 = get_purity_representations(confusion2)
    plot_confusion(confusion_pur1, config_train["classes"],
                   path_template.format(args.fold, "purity1"), "Purity")
    plot_confusion(confusion_pur2, config_train["classes"],
                   path_template.format(args.fold, "purity2"), "Purity")
    plot_confusion(confusion_pur3, config_train["classes"],
                   path_template.format(args.fold, "purity_cw1"), "Purity")
    plot_confusion(confusion_pur4, config_train["classes"],
                   path_template.format(args.fold, "purity_cw2"), "Purity")


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
