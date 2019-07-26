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
from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.models import load_model

sys.path.append("htt-ml/training")

import logging
logger = logging.getLogger("keras_significance")
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

def get_efficiency_dict(purity_dict, weight_dict, classes):
    efficiency_dict = OrderedDict()
    efficiency_weights = OrderedDict()
    for class_ in classes:
        true_dict = OrderedDict()
        true_weights = OrderedDict()
        efficiency_dict[class_] = true_dict
        efficiency_weights[class_] = true_weights
        # for class_2 in classes:
        #     efficiency_dict[class_][class_2] = list()
        #     efficiency_weights[class_][class_2] = list()

    for predicted_class, predicted_dict in purity_dict.items():
        predicted_weights = weight_dict[predicted_class]
        for true_class, true_dict in predicted_dict.items():
            true_weights = predicted_weights[true_class]
            try:
                assert len(true_dict) == len(true_weights)
            except AssertionError as e:
                logger.critical('{}: Length {} is not length {}'.format(e, len(true_dict), len(true_weights)))
            efficiency_dict[true_class][predicted_class] = true_dict
            efficiency_weights[true_class][predicted_class] = true_weights
    return efficiency_dict, efficiency_weights

def calculate_metric_per_cut(key, item, classes, weights, output_path, cuts, metric_name):
    bins = np.linspace(0,1,cuts+1)
    all_metrics = []
    for cut in bins:
        sum_of_all = 0
        sum_of_target = 0
        for class_ in classes:
            sum_of_class = 0
            for i, element in enumerate(item[class_]):
                if element > cut:
                    sum_of_class += weights[class_][i]
            if class_ == key:
                sum_of_target = sum_of_class
            sum_of_all += sum_of_class
        if sum_of_all != 0.0:
            all_metrics.append(sum_of_target/sum_of_all)
        else:
            #print('Sum of all is {} for cut {}'.format(sum_of_all,cut))
            all_metrics.append(0)

    plt.figure(figsize=(6,6))
    plt.plot(bins, all_metrics, 'ro')
    plt.xlim([0.,1.0])
    plt.ylim([0.0,1.1])
    plt.ylabel(metric_name)
    plt.xlabel(r'Cut')
    plt.title('{} for class {}'.format(metric_name, key))
    plt.tight_layout()
    plt.savefig(output_path + ".png", bbox_inches="tight")
    plt.savefig(output_path + ".pdf", bbox_inches="tight")
    plt.clf()



def plot_significance_per_bin(title, dictionary, classes, weights, output_path, nbins = 20):
    list_of_histograms = []
    list_of_weights = []
    signal_index = classes.index(title)
    bins = np.linspace(0,1,nbins + 1)
    for key, item in dictionary.items():
        list_of_histograms.append(item)
    for key, item in weights.items():
        list_of_weights.append(item)
    fig, (ax, ax_2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(10,10))
    try:
        hist_values, _, _ = ax.hist(list_of_histograms, bins=bins, stacked=True, weights=list_of_weights, normed=False, label=classes)#, log=True)
    except Exception as e:
        logger.exception(e)
        return

    if signal_index == 0:
        signal_list = hist_values[signal_index]
    else:
        signal_list = hist_values[signal_index] - hist_values[signal_index-1]
    significance_per_bin = []
    x_values = []
    for i, bin in enumerate(hist_values[-1]):
        signal_value = signal_list[i]
        background_value = bin
        significance_per_bin.append(signal_value/np.sqrt(background_value))
        x_values.append(float(i)/float(nbins) + 1./float(nbins)/2.)

    ax.legend()
    ax.set_xlim([0.,1.0])
    ax.set_title("Class {} significance".format(title))
    #ax.set_yscale('log')
    plt.xlabel('Neural Network Score'), ax.set_ylabel("Weighted # of events")

    ax_2.plot(x_values, significance_per_bin, 'ro')
    ax_2.set_xlim([0.,1.0])
    ax_2.set_ylim([0.,100.])
    ax_2.set_ylabel(r'$S/\sqrt{S+B}$')
    fig.tight_layout()
    fig.savefig(output_path + ".png", bbox_inches="tight")
    fig.savefig(output_path + ".pdf", bbox_inches="tight")

    logger.info("Saved results to {}".format(output_path))
    fig.clf()
    plt.clf()

def plot_significance_per_cut(title, dictionary, classes, weights, output_path, nbins = 20):
    list_of_histograms = []
    list_of_weights = []
    signal_index = classes.index(title)
    bins = np.linspace(0,1,nbins + 1)
    for key, item in dictionary.items():
        list_of_histograms.append(item)
    for key, item in weights.items():
        list_of_weights.append(item)
    fig, (ax, ax_2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(10,10))
    try:
        hist_values, _, _ = ax.hist(list_of_histograms, bins=bins, stacked=True, weights=list_of_weights, normed=False, label=classes)#, log=True)
    except Exception as e:
        logger.exception(e)
        return

    if signal_index == 0:
        signal_list = hist_values[signal_index]
    else:
        signal_list = hist_values[signal_index] - hist_values[signal_index-1]
    significance_per_cut = []
    #logger.info('Printing signal list and hist values for class {}'.format(title))
    #print(signal_list, hist_values)
    for i, bin in enumerate(bins):
        signal_value = sum(signal_list[i:])
        background_value = sum(hist_values[-1][i:])
        #print('Cut {}: Signal Value {} - Background value {}'.format(bin, signal_value, background_value))
        significance_per_cut.append(signal_value/np.sqrt(background_value))

    ax.legend()
    ax.set_xlim([0.,1.0])
    ax.set_title("Class {} significance per cut".format(title))
    #ax.set_yscale('log')
    ax.set_xlabel('Neural Network Score'), ax.set_ylabel("Weighted # of events")

    ax_2.plot(bins, significance_per_cut, 'ro')
    ax_2.set_xlim([0.,1.0])
    ax_2.set_ylim([0., 110.])
    ax_2.set_ylabel(r'$S/\sqrt{S+B}$')
    ax_2.set_xlabel(r'Cut')
    fig.tight_layout()
    fig.savefig(output_path + ".png", bbox_inches="tight")
    fig.savefig(output_path + ".pdf", bbox_inches="tight")

    logger.info("Saved results to {}".format(output_path))
    fig.clf()
    plt.clf()

def main(args, config_test, config_train):
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.info("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.info("Load keras model %s.", path)
    model = load_model(path, compile=False)
                       #, custom_objects={'focal_loss_fixed': focal_loss()})

    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.info("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)

    # Setup dictionaries

    all_classes = OrderedDict()
    all_weights = OrderedDict()
    for class_ in config_train['classes']:
        predicted_dict = OrderedDict()
        predicted_weights = OrderedDict()
        all_classes[class_] = predicted_dict
        all_weights[class_] = predicted_weights
        for class_2 in config_train['classes']:
            all_classes[class_][class_2] = []
            all_weights[class_][class_2] = []

    classes = config_train['classes']
    class_weights = config_train["class_weights"]
    for i_class, class_ in enumerate(config_train["classes"]):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception
        friend_trees_names = [k.GetName() for k in file_.GetListOfKeys() if k.GetName().startswith("_".join([class_,"friend"]))]
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

        weight = array("f", [-999])
        tree.SetBranchAddress(config_test["weight_branch"], weight)

        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)
            values_stacked = np.hstack(values).reshape(1, len(values))
            values_preprocessed = preprocessing.transform(values_stacked)
            response = model.predict(values_preprocessed)
            if len(response) == 3:
                response = response[0]
            response = np.squeeze(response)
            max_index = np.argmax(response)
            max_value = np.max(response)
            predicted_class = classes[max_index]
            all_classes[predicted_class][class_].append(max_value)
            all_weights[predicted_class][class_].append(weight[0]*class_weights[class_])

    for key, item in all_classes.items():
        logger.debug('Getting histogram of class {}'.format(key))
        weight_dictionary = all_weights[key]
        output_path = os.path.join(config_train['output_path'], 'fold{}_signal_strength_per_bin_{}'.format(args.fold, key))
        plot_significance_per_bin(key, item, classes=classes, weights = weight_dictionary, output_path=output_path, nbins=30)
        output_path = os.path.join(config_train['output_path'], 'fold{}_signal_strength_per_cut_{}'.format(args.fold, key))
        plot_significance_per_cut(key, item, classes=classes, weights = weight_dictionary, output_path=output_path, nbins=30)

    efficiency_dict, efficiency_weight = get_efficiency_dict(purity_dict=all_classes, weight_dict=all_weights, classes=classes)

    for key, item in all_classes.items():
        logger.debug('Calculating purity per cut of class {}'.format(key))
        weight_dictionary = all_weights[key]
        output_path = os.path.join(config_train['output_path'],
                                   'fold{}_purity_per_cut_{}'.format(args.fold, key))
        calculate_metric_per_cut(key, item, classes=classes, weights = weight_dictionary, output_path=output_path, cuts = 30, metric_name="Purity")

    for key, item in efficiency_dict.items():
        logger.debug('Calculating efficiency per cut of class {}'.format(key))
        weight_dictionary = efficiency_weight[key]
        output_path = os.path.join(config_train['output_path'],
                                   'fold{}_efficiency_per_cut_{}'.format(args.fold, key))
        calculate_metric_per_cut(key, item, classes=classes, weights = weight_dictionary, output_path=output_path, cuts = 30, metric_name="Efficiency")

if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)