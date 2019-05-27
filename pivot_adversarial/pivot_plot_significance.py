#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser


import argparse
from array import array
import yaml
import pickle
import numpy as np
import os
import glob
import matplotlib as mpl
import time
from collections import OrderedDict

mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt

from keras.models import load_model

import logging
logger = logging.getLogger("pivot_plot_significance")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot a lot of stuff.")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_testing", help="Path to testing config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))


def plot_histograms(title, dictionary, classes, weights, output_path, nbins = 20):
    list_of_histograms = []
    list_of_backgrounds = []
    list_of_background_weights = []
    list_of_weights = []
    #signal_index = classes.index(title)
    shifted_histograms = dict()
    shifted_weights = dict()
    bins = np.linspace(0,1,nbins + 1)
    for key, item in dictionary.items():
        if key == 'ggh':
            for key, list in item.items():
                if key == 'nom':
                    list_of_histograms.append(list)
                else:
                    shifted_histograms[key] = list
        else:
            list_of_backgrounds.append(item)
    for key, item in weights.items():
        if key == 'ggh':
            for key, list in item.items():
                if key == 'nom':
                    list_of_weights.append(list)
                else:
                    shifted_weights[key] = list
        else:
            list_of_background_weights.append(item)
    fig, (ax, ax_2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(10,10))

    flat_list_of_backgrounds = [item for sublist in list_of_backgrounds for item in sublist]
    flat_list_of_background_weights = [item for sublist in list_of_background_weights for item in sublist]
    list_of_histograms.append(flat_list_of_backgrounds)
    list_of_weights.append(flat_list_of_background_weights)

    try:
        hist_values, _, _ = ax.hist(list_of_histograms, bins=bins, stacked=True, histtype='step', lw=3, weights=list_of_weights, normed=False, label=['ggh_nom', 'background'], log=True)
        hist_values_up, _, _ = ax.hist(shifted_histograms['up'], bins=bins, stacked=False, histtype='step', lw=3, weights=shifted_weights['up'], normed=False, label='ggh_up', log=True)
        hist_values_down, _, _ = ax.hist(shifted_histograms['down'], bins=bins, stacked=False, histtype='step', lw=3, weights=shifted_weights['down'], normed=False, label='ggh_down', log=True)#, log=True)
    except Exception as e:
        logger.exception(e)
        return

    signal_list_nom = hist_values[0]
    significance_per_bin = []
    x_values = []
    for i, bin in enumerate(hist_values[-1]):
        signal_value = signal_list_nom[i]
        signal_sigma = np.abs(hist_values_up[i] - hist_values_down[i] / 2.)
        background_value = bin
        significance_per_bin.append(signal_value/np.sqrt(background_value + signal_sigma))
        x_values.append(float(i)/float(nbins) + 1./float(nbins)/2.)
        print('Up-Ratio of bin {}: {}'.format(i,hist_values_up[i]/signal_value))
        print('Down-Ratio of bin {}: {}'.format(i, hist_values_down[i] / signal_value))

    ax.legend()
    ax.set_xlim([0.,1.0])
    ax.set_title("Class {} significance".format(title))
    #ax.set_yscale('log')
    plt.xlabel('Neural Network Score'), ax.set_ylabel("Weighted # of events")

    ax_2.plot(x_values, significance_per_bin, 'ro')
    ax_2.set_xlim([0.,1.0])
    ax_2.set_ylim([0.9,1.1])
    ax_2.set_ylabel(r'$S/\sqrt{S+B + \sigma_S}$')
    fig.tight_layout()
    fig.savefig(output_path + ".png", bbox_inches="tight")
    fig.savefig(output_path + ".pdf", bbox_inches="tight")

    logger.info("Saved results to {}".format(output_path))
    fig.clf()
    plt.clf()
    pass

def plot_ratios(title, dictionary, classes, weights, output_path, nbins = 20):
    list_of_histograms = []
    list_of_backgrounds = []
    list_of_background_weights = []
    list_of_weights = []
    #signal_index = classes.index(title)
    shifted_histograms = dict()
    shifted_weights = dict()
    bins = np.linspace(0,1,nbins + 1)
    for key, item in dictionary.items():
        if key == 'ggh':
            for key, list in item.items():
                if key == 'nom':
                    list_of_histograms.append(list)
                else:
                    shifted_histograms[key] = list
        else:
            list_of_backgrounds.append(item)
    for key, item in weights.items():
        if key == 'ggh':
            for key, list in item.items():
                if key == 'nom':
                    list_of_weights.append(list)
                else:
                    shifted_weights[key] = list
        else:
            list_of_background_weights.append(item)
    fig, (ax, ax_2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(10,10))

    flat_list_of_backgrounds = [item for sublist in list_of_backgrounds for item in sublist]
    flat_list_of_background_weights = [item for sublist in list_of_background_weights for item in sublist]
    list_of_histograms.append(flat_list_of_backgrounds)
    list_of_weights.append(flat_list_of_background_weights)

    try:
        hist_values, _, _ = ax.hist(list_of_histograms, bins=bins, stacked=True, histtype='step', lw=3, weights=list_of_weights, normed=False, label=['ggh_nom', 'background'], log=0)
        hist_values_up, _, _ = ax.hist(shifted_histograms['up'], bins=bins, stacked=False, histtype='step', lw=3, weights=shifted_weights['up'], normed=False, label='ggh_up', log=0)
        hist_values_down, _, _ = ax.hist(shifted_histograms['down'], bins=bins, stacked=False, histtype='step', lw=3, weights=shifted_weights['down'], normed=False, label='ggh_down', log=0)#, log=True)
    except Exception as e:
        logger.exception(e)
        return

    signal_list_nom = hist_values[0]
    ratio_up_per_bin = []
    ratio_down_per_bin = []
    x_values = []
    for i, bin in enumerate(hist_values[-1]):
        signal_value = signal_list_nom[i]
        signal_sigma = np.abs(hist_values_up[i] - hist_values_down[i] / 2.)
        background_value = bin
        #significance_per_bin.append(signal_value/np.sqrt(background_value + signal_sigma))
        x_values.append(float(i)/float(nbins) + 1./float(nbins)/2.)
        ratio_up_per_bin.append(hist_values_up[i]/signal_value)
        ratio_down_per_bin.append(hist_values_down[i]/signal_value)
        print('Up-Ratio of bin {}: {}'.format(i,hist_values_up[i]/signal_value))
        print('Down-Ratio of bin {}: {}'.format(i, hist_values_down[i] / signal_value))

    ax.legend()
    ax.set_xlim([0.,1.0])
    ax.set_title("Class {} significance".format(title))
    #ax.set_yscale('log')
    plt.xlabel('Neural Network Score'), ax.set_ylabel("Weighted # of events")

    ax_2.plot(x_values, ratio_up_per_bin, 'ro', label='Up ratio', color='green')
    ax_2.plot(x_values, ratio_down_per_bin, 'ro', label='Down ratio', color='red')
    ax_2.axhline(y=1.0, color='grey', linestyle='--')
    ax_2.legend(loc="upper left")
    ax_2.set_xlim([0.,1.0])
    ax_2.set_ylim([0.9,1.1])
    ax_2.set_ylabel(r'Ratio Shift/Nominal')
    fig.tight_layout()
    fig.savefig(output_path + ".png", bbox_inches="tight")
    fig.savefig(output_path + ".pdf", bbox_inches="tight")

    logger.info("Saved results to {}".format(output_path))
    fig.clf()
    plt.clf()
    pass

def main(args, config_test, config_train):
    lam = config_train['model']['lambda']
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.info("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    classifier_dict = {'pre': None, 'decorrelated': None}

    classifier_dict['pre'] = os.path.join(config_train["output_path"],
                        config_test["pre_classifier_model"][args.fold].format(lam))
    classifier_dict['decorrelated'] = os.path.join(config_train["output_path"],
                        config_test["decorrelated_classifier_model"][args.fold].format(lam))
    for model_key, model_path in classifier_dict.items():
        logger.info("Load keras model %s.", model_path)
        model = load_model(model_path, compile=False)

        path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
        logger.info("Loop over test dataset %s to get model response.", path)
        file_ = ROOT.TFile(path)

        # Setup dictionaries

        all_classes = OrderedDict()
        all_weights = OrderedDict()
        for class_predicted in config_train['classes']:
            actual_dict = OrderedDict()
            actual_weights = OrderedDict()
            all_classes[class_predicted] = actual_dict
            all_weights[class_predicted] = actual_weights
            for actual_class in config_train['classes']:
                if class_predicted == 'ggh' and actual_class=='ggh':
                    shifted_ggh = OrderedDict()
                    weights_ggh = OrderedDict()
                    all_classes[class_predicted][actual_class] = shifted_ggh
                    all_weights[class_predicted][actual_class] = weights_ggh
                    for shift in ['nom', 'up', 'down']:
                        all_classes[class_predicted][actual_class][shift] = []
                        all_weights[class_predicted][actual_class][shift] = []
                else:
                    all_classes[class_predicted][actual_class] = []
                    all_weights[class_predicted][actual_class] = []

        classes = config_train['classes']
        class_weights = config_train["class_weights"]
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
            uncertainty = array('f', [-999])
            tree.SetBranchAddress(config_test['target_uncertainty'], uncertainty)

            for i_event in range(tree.GetEntries()):
                tree.GetEntry(i_event)

                values_stacked = np.hstack(values).reshape(1, len(values))
                values_nom = preprocessing.transform(values_stacked)
                response_nom = np.squeeze(model.predict(values_nom))
                max_index = np.argmax(response_nom)
                max_value = np.max(response_nom)
                predicted_class = classes[max_index]
                if predicted_class == 'ggh' and class_ == 'ggh':
                    all_classes[predicted_class][class_]['nom'].append(max_value)
                    all_weights[predicted_class][class_]['nom'].append(weight[0] * class_weights[class_] / 3.)
                else:
                    all_classes[predicted_class][class_].append(max_value)
                    all_weights[predicted_class][class_].append(weight[0] * class_weights[class_])

                if class_ == 'ggh':
                    values_stacked_up = values_stacked * uncertainty[0]*1.5
                    values_stacked_down = values_stacked * 1./(float(uncertainty[0]*1.5))
                    print(values_stacked, values_stacked_up, values_stacked_down)
                    values_up = preprocessing.transform(values_stacked_up)
                    values_down = preprocessing.transform(values_stacked_down)
                    response_down = np.squeeze(model.predict(values_down))
                    response_up = np.squeeze(model.predict(values_up))
                    max_index_down = np.argmax(response_down)
                    max_value_down = np.max(response_down)
                    predicted_class_down = classes[max_index_down]
                    if predicted_class_down == 'ggh':
                        all_classes[predicted_class_down][class_]['down'].append(max_value_down)
                        all_weights[predicted_class_down][class_]['down'].append(weight[0] * class_weights[class_] / 3.)
                    max_index_up = np.argmax(response_up)
                    max_value_up = np.max(response_up)
                    predicted_class_up = classes[max_index_up]
                    if predicted_class_up == 'ggh':
                        all_classes[predicted_class_up][class_]['up'].append(max_value_up)
                        all_weights[predicted_class_up][class_]['up'].append(weight[0] * class_weights[class_] / 3.)

        for class_key, class_ in all_classes.items():
            logger.debug('Getting histogram of class {}'.format(class_key))
            if class_key == 'ggh':
                weight_dictionary = all_weights[class_key]
                #labels = ['ggh_nom', 'ggh_up', 'ggh_down', 'qqh', 'ztt', 'zll', 'w', 'tt', 'ss', 'misc']
                output_path = os.path.join(config_train['output_path'], 'fold{}_ratios_{}_{}_l={}'.format(args.fold, model_key, class_key, lam))
                plot_ratios(class_key, class_, classes=classes, weights = weight_dictionary, output_path=output_path, nbins=30)


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
