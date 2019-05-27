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

def plot(p,y,weights_up, weights_down, path):
    # Plot shifts
    plt.figure(figsize=(6, 6))
    bins = np.linspace(0, 1, 100)
    plt.hist(p, bins=bins, histtype="step", lw=3, label="Nominal")
    plt.hist(p, bins=bins, weights=weights_up, histtype="step", lw=3, label="Up")
    plt.hist(
        p, bins=bins, weights=weights_down, histtype="step", lw=3, label="Down")
    plt.legend()
    plt.xlim((0, 1))
    plt.xlabel("NN output")
    plt.savefig(path + "/test_decorrelated.png")
    plt.clf()

    # Plot splitted in signal and background
    plt.figure(figsize=(6, 6))
    plt.hist(p[y == 1], bins=bins, histtype="step", lw=3, label="Signal")
    plt.hist(p[y == 0], bins=bins, histtype="step", lw=3, label="Background")
    plt.hist(
        p[y == 0],
        weights=weights_up[y == 0],
        bins=bins,
        histtype="step",
        lw=3,
        label="Background (Up)")
    plt.hist(
        p[y == 0],
        weights=weights_down[y == 0],
        bins=bins,
        histtype="step",
        lw=3,
        label="Background (Down)")
    plt.legend()
    plt.xlim((0, 1))
    plt.xlabel("NN output")
    plt.savefig(path + "/separation_decorrelated.png")

def main(args, config_train):
    output_path = config_train['output_path']
    path = os.path.join(output_path, 'test.pickle')
    x, y, weights_up, weights_down = pickle.load(open(path, "rb"))

    path = os.path.join(output_path, 'toy_model.h5')
    model = load_model(path, compile=False)

    p = model.predict(x).squeeze()
    plot(p, y, weights_up, weights_down, output_path)

if __name__ == "__main__":
    args = parse_arguments()
    config_train = parse_config(args.config_training)
    main(args, config_train)