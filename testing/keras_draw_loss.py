#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
import yaml
import pickle
import os
import re
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import logging
logger = logging.getLogger("keras_recall_precision_score")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)




def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate recall and precision.")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    return parser.parse_args()

def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))

def draw_plots(variable_name, history, y_label):

    # NOTE: Matplotlib needs to be imported after Keras/TensorFlow because of conflicting libraries
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt

    plt.clf()

    epochs = range(1, len(history[variable_name]) + 1)
    plt.plot(epochs, history[variable_name], lw=3, label="Train {}".format(variable_name))
    plt.plot(
        epochs, history["val_{}".format(variable_name)], lw=3, label="Validation {}".format(variable_name))
    plt.xlabel("Epoch"), plt.ylabel(y_label)
    path_plot = os.path.join(config_train["output_path"],
                             "fold{}_{}".format(args.fold, variable_name))
    plt.legend()
    plt.savefig(path_plot+".png", bbox_inches="tight")
    plt.savefig(path_plot+".pdf", bbox_inches="tight")


def draw_validation_losses(variable_names, history, y_label, class_names):

    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    j=0
    for variable_name in variable_names:
        epochs = range(1, len(history[variable_name]) + 1)
        if 'significance_per_bin' in variable_name:
            digits = re.findall(r'\d', variable_name)
            if digits:
                digit = int(digits[0])
                ax1.plot(
                    epochs, history["{}".format(variable_name)], lw=3, label="Val {}".format(class_names[digit]))
            else:
                ax1.plot(
                    epochs, history["{}".format(variable_name)], lw=3, label="Val {}".format(class_names[0]))
        elif 'loss_ce' in variable_name:
            ax2.plot(
                epochs, np.asarray(history["{}".format(variable_name)])*10., lw=3,
                label="Val {}".format(variable_name), color='y')
    plt.xlabel("Epoch")
    ax1.set_ylabel('Significance-' + y_label)
    ax2.set_ylabel('CE-' + y_label )
    ax2.tick_params(axis='y', labelcolor='y')
    path_plot = os.path.join(config_train["output_path"],
                             "fold{}_{}".format(args.fold, 'val_loss'))
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")
    fig.clf()

def draw_training_losses(variable_names, history, y_label, class_names):

    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    j=0
    for variable_name in variable_names:
        epochs = range(1, len(history[variable_name]) + 1)
        if 'significance_per_bin' in variable_name:
            digits = re.findall(r'\d', variable_name)
            if digits:
                digit = int(digits[0])
                ax1.plot(
                    epochs, history["{}".format(variable_name)], lw=3, label="Train {}".format(class_names[digit]))
            else:
                ax1.plot(
                    epochs, history["{}".format(variable_name)], lw=3, label="Train {}".format(class_names[0]))
        elif 'loss_ce' in variable_name:
            #print(history["{}".format(variable_name)])
            ax2.plot(
                epochs, np.asarray(history["{}".format(variable_name)])*10., lw=3,
                label="Train {}".format(variable_name), color='y')
    plt.xlabel("Epoch")
    ax1.set_ylabel('Significance-' + y_label)
    ax2.set_ylabel('CE-' + y_label )
    ax2.tick_params(axis='y', labelcolor='y')
    path_plot = os.path.join(config_train["output_path"],
                             "fold{}_{}".format(args.fold, 'train_loss'))
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")
    fig.clf()


def draw_significance_per_class(metric_data, metric_names, classes, y_labels):
    epochs = range(1, len(metric_data) + 1)

    list_of_metrics = dict()
    for metric_name in metric_names:
        list_of_metrics[metric_name] = dict()
        for class_ in classes:
            list_of_metrics[metric_name][class_] = []

    for data_point in metric_data:
        for metric_name in metric_names:
            for i_class, class_ in enumerate(classes):
                list_of_metrics[metric_name][class_].append(data_point[metric_name][str(i_class)])
    for metric_name, label_name in zip(metric_names, y_labels):
        plt.clf()
        for class_ in classes:
            plt.plot(epochs, list_of_metrics[metric_name][class_], lw=3, label="{}".format(class_))
        plt.xlabel("Epoch"), plt.ylabel(label_name)
        #plt.title("Validation score for {}".format(label_name))
        path_plot = os.path.join(config_train["output_path"],
                                 "fold{}_{}".format(args.fold, metric_name))
        plt.legend()
        plt.savefig(path_plot+".png", bbox_inches="tight")
        plt.savefig(path_plot+".pdf", bbox_inches="tight")


def main(args, config_train):
    path = os.path.join(config_train["output_path"],
                        "fold{}_keras_history.pickle".format(args.fold))
    logger.debug("Load history %s.", path)
    history_dict, metrics_data = pickle.load(open(path, "rb"))

    classes = config_train['classes']

    variable_names_train = []
    variable_names_val = []
    for variable_name in history_dict.keys():
        # if 'loss_ce' in variable_name:
        #     continue
        if 'val' in variable_name:
            variable_names_val.append(variable_name)
        else:
            variable_names_train.append(variable_name)

    variable_names_train = sorted(variable_names_train)
    variable_names_val = sorted(variable_names_val)

    draw_validation_losses(variable_names_val,history_dict, y_label='Loss', class_names = classes)
    draw_training_losses(variable_names_train, history_dict, y_label='Loss', class_names = classes)

    draw_plots(variable_name="loss", history=history_dict, y_label="Loss")
    draw_significance_per_class(metric_data=metrics_data, metric_names=['Significance'], classes=classes, y_labels='Loss')


if __name__ == "__main__":
    args = parse_arguments()
    config_train = parse_config(args.config_training)
    main(args, config_train)
