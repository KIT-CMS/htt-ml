#!/usr/bin/env python
import argparse
import yaml
import pickle
import numpy as np
import os
import glob
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

import logging
logger = logging.getLogger("pivot_plotting")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate AMS score.")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_testing", help="Path to testing config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))


def get_AMS1(pred, Y, Z):
    x00 = np.sort(pred[((Y == 0) * (Z == 0))].flatten())
    x01 = np.sort(pred[((Y == 0) * (Z == 1))].flatten())
    x10 = np.sort(pred[((Y == 1) * (Z == 0))].flatten())

    n_points = 100
    AMS1 = np.zeros(n_points)
    AMS = np.zeros(n_points)
    all_ns = np.zeros(n_points)
    all_nb = np.zeros(n_points)
    all_nb1 = np.zeros(n_points)
    all_sigb = np.zeros(n_points)

    cuts = np.zeros(n_points)
    sig_eff = np.zeros(n_points)
    ns_tot = x10.shape[0]

    for i, c_i in enumerate(np.linspace(0.0, 1.0, n_points)):
        cuts[i] = c_i
        ns = (100. / x10.size) * np.count_nonzero(x10 > c_i)
        nb = (1000. / x00.size) * np.count_nonzero(x00 > c_i)
        nb1 = (1000. / x01.size) * np.count_nonzero(x01 > c_i)
        sig_b = 1.0 * np.abs(nb - nb1)

        b0 = 0.5 * (nb - sig_b ** 2 + ((nb - sig_b ** 2) ** 2 + 4 * (ns + nb) * (sig_b ** 2)) ** 0.5)
        AMS1[i] = (2.0 * ((ns + nb) * np.log((ns + nb) / b0) - ns - nb + b0) + ((nb - b0) / sig_b) ** 2) ** 0.5
        AMS[i] = (2*((ns + nb + 10.) * np.log(1 + ns/(nb + 10.)) - ns))**0.5

        all_ns[i] = ns
        all_nb[i] = nb
        all_nb1[i] = nb
        all_sigb[i] = sig_b

        sig_eff[i] = (1.0 * ns) / ns_tot

    return cuts, AMS

def main(args, config_test, config_train):

    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    #logger.debug("Load preprocessing %s.", path)
    #preprocessing = pickle.load(open(path, "rb"))

    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold].format(config_train['model']['lambda']))
    logger.debug("Load keras model %s.", path)
    model = load_model(path, compile=False)

    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.debug("Loop over test dataset %s to get model response.", path)
    with open(path, mode='rb') as fd:
        X_train, X_test, y_train, y_test, z_train, z_test, scaler = pickle.load(fd)

    cuts, a = get_AMS1(model.predict(X_test), y_test, z_test[:,1])


    plt.plot(cuts, a, label=r"$\lambda={}$".format(config_train['model']['lambda']))

    output_path = os.path.join(config_train['output_path'], 'ams-l={}.png'.format(config_train['model']['lambda']))

    plt.legend(loc="best")
    plt.ylabel("AMS")
    plt.xlabel("threshold on $f(X)$")
    plt.grid()
    plt.savefig(output_path)

if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
