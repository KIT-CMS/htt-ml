#!/usr/bin/env python
import argparse
import yaml
import pickle
import numpy as np
import matplotlib as mpl
import os

mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

import logging
logger = logging.getLogger("plotting")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate AMS score.")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))

def plot(p_label, w, w_up, w_down, path):
    # Plot shifts
    #plt.figure(figsize=(6, 6))
    fig, (ax, ax_2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 6))
    bins = np.linspace(0, 1, 50)
    stacked_values, _, _ =ax.hist(p_label, bins=bins, stacked=True, weights=w, histtype="step", lw=3, label=["Nominal Signal", "Nominal Background"])
    up_values, _, _ = ax.hist(p_label[0], bins=bins, weights=w_up[0]*w[0], histtype="step", lw=3, label="Up Signal")
    down_values, _, _ = ax.hist(
        p_label[0], bins=bins, weights=w_down[0]*w[0], histtype="step", lw=3, label="Down Signal")
    ax.legend()
    plt.xlim((0, 1))
    plt.xlabel("NN output")
    plt.title("ggH output")

    signal_list_nom = stacked_values[0]
    ratio_up_per_bin = []
    ratio_down_per_bin = []
    x_values = []
    for i, bin in enumerate(stacked_values[-1]):
        signal_value = signal_list_nom[i]
        x_values.append(float(i)/float(50) + 1./float(50)/2.)
        ratio_up_per_bin.append(up_values[i]/signal_value)
        ratio_down_per_bin.append(down_values[i]/signal_value)

    ax_2.plot(x_values, ratio_up_per_bin, 'ro', label='Up ratio', color='green')
    ax_2.plot(x_values, ratio_down_per_bin, 'ro', label='Down ratio', color='red')
    ax_2.axhline(y=1.0, color='grey', linestyle='--')
    ax_2.legend(loc="upper right")
    ax_2.set_xlim([0.,1.0])
    ax_2.set_ylim([0.9,1.1])
    ax_2.set_ylabel(r'Ratio Shift/Nominal')
    fig.tight_layout()
    fig.savefig(path + ".png", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")

    logger.info("Saved results to {}".format(path))
    fig.clf()
    plt.clf()
    pass



def get_class_data(p, y, w, w_up, w_down, label):
    def get_weights(weights, mask, argmax):
        w_mask = weights[mask]
        w_signal = w_mask[argmax == label]
        w_background = w_mask[argmax != label]
        w_signal = np.reshape(w_signal, w_signal.shape[0])
        w_background = np.reshape(w_background, w_background.shape[0])
        return [w_signal, w_background]

    mask = (y[:, label] == 1)
    p_label = p[mask]
    p_argmax = np.argmax(p_label, axis=-1)
    p_label_true = p_label[p_argmax == label]
    p_label_background = p_label[p_argmax != label]
    #prediction = [np.max(p_label_true, axis=-1), np.max(p_label_background, axis=-1)]
    p_label = [np.max(p_label_true, axis=-1), np.max(p_label_background, axis=-1)]
    w_label = get_weights(w, mask, p_argmax)
    w_up_label = get_weights(w_up, mask, p_argmax)
    w_down_label = get_weights(w_down, mask, p_argmax)
    return p_label, w_label, w_up_label, w_down_label


def main(args, config_train):

    path = os.path.join(config_train["pickle_train"][(1, 0)[args.fold]])
    logger.info("Make inference over dataset {}.".format(path))
    x, y, weights, weights_up, weights_down = pickle.load(open(path, "rb"))

    output_path = config_train['output_path']
    path = os.path.join(output_path, config_train['model']['name']+'_fold{}.h5'.format(args.fold))
    model = load_model(path, compile=False)

    logger.info("Get prediction...")

    p = model.predict(x)

    p_label, w, w_up, w_down = get_class_data(p, y, weights, weights_up, weights_down, label=0)

    logger.info("Getting plots...")

    path = os.path.join(output_path, config_train['model']['name']+ '_test_fold{}'.format(args.fold))

    plot(p_label, w, w_up, w_down, path)

if __name__ == "__main__":
    args = parse_arguments()
    config_train = parse_config(args.config_training)
    main(args, config_train)