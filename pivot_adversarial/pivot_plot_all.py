#!/usr/bin/env python
import argparse
import yaml
import pickle
import numpy as np
import os
import glob
import matplotlib as mpl
from sklearn.metrics import roc_curve

mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model
import h5py

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

def get_data(scaler):
    # Get data from http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/highlevel/

    f = h5py.File("/home/sjoerger/workspace/pivot_adversarials/jet_example/test_no_pile_5000000.h5", "r")
    X_no_pile = f["features"].value
    y_no_pile = f["targets"].value.ravel()

    f = h5py.File("/home/sjoerger/workspace/pivot_adversarials/jet_example/test_pile_5000000.h5", "r")
    X_pile = f["features"].value
    y_pile = f["targets"].value.ravel()

    X = np.vstack((X_no_pile, X_pile))
    y = np.concatenate((y_no_pile, y_pile)).ravel()
    z = np.zeros(len(X))
    z[len(X_no_pile):] = 1

    strates = np.zeros(len(X))
    strates[(y == 0) & (z == 0)] = 0
    strates[(y == 0) & (z == 1)] = 1
    strates[(y == 1) & (z == 0)] = 2
    strates[(y == 1) & (z == 1)] = 3

    from keras.utils import np_utils
    z = np_utils.to_categorical(z.astype(np.int))

    X = scaler.transform(X)

    return X,y,z


def get_AMS1(pred, Y, Z):
    x00 = np.sort(pred[((Y == 0) * (Z == 0))].flatten())
    x01 = np.sort(pred[((Y == 0) * (Z == 1))].flatten())
    x10 = np.sort(pred[((Y == 1) * (Z == 0))].flatten())

    # x0 = np.sort(pred[(Y==0)]).flatten()
    # x1 = np.sort(pred[(Y==1)]).flatten()

    n_points = 100
    AMS1 = np.zeros(n_points)
    #AMS = np.zeros(n_points)
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

        # ns = (100./x1.size) * np.count_nonzero(x1 > c_i)
        # nb = (1000./x0.size) * np.count_nonzero(x0 > c_i)

        b0 = 0.5 * (nb - sig_b ** 2 + ((nb - sig_b ** 2) ** 2 + 4 * (ns + nb) * (sig_b ** 2)) ** 0.5)
        AMS1[i] = (2.0 * ((ns + nb) * np.log((ns + nb) / b0) - ns - nb + b0) + ((nb - b0) / sig_b) ** 2) ** 0.5
        #AMS[i] = (2*((ns + nb + 10.) * np.log(1 + ns/(nb + 10.)) - ns))**0.5

        all_ns[i] = ns
        all_nb[i] = nb
        all_nb1[i] = nb
        all_sigb[i] = sig_b

        sig_eff[i] = (1.0 * ns) / ns_tot

    return cuts, AMS1

def main(args, config_test, config_train):
    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.debug("Loop over test dataset %s to get model response.", path)
    with open(path, mode='rb') as fd:
        X_train, X_test, y_train, y_test, z_train, z_test, scaler = pickle.load(fd)

    X,y,z = get_data(scaler)

    logger.debug('Got testdata...')

    indices = np.random.permutation(len(X))
    indices = indices[:5000000]

    fpr = dict()
    tpr = dict()

    for lam in [0,1,10,500]:
        path = os.path.join(config_train['output_path'],'fold0_classifier-l={}.h5'.format(lam))
        logger.debug("Load keras model %s.", path)
        model = load_model(path, compile=False)

        predictions = model.predict(X[indices])

        cuts, a = get_AMS1(predictions, y[indices], z[indices,1])

        plt.plot(cuts, a, label=r"$\lambda={}$".format(lam))

        fpr[lam], tpr[lam], _ = roc_curve(y[indices], predictions)

    output_path = os.path.join(config_train['output_path'], 'ams-all'.format(config_train['model']['lambda']))

    plt.legend(loc="best")
    plt.ylabel("AMS")
    plt.xlabel("threshold on $f(X)$")
    plt.grid()
    plt.savefig(output_path + '.png')
    plt.savefig(output_path + '.pdf')


    plt.clf()

    for key, item in fpr.items():
        fpr_lam = item
        tpr_lam = tpr[key]
        plt.plot(fpr_lam, tpr_lam, lw=2, label=r'$\lambda={}$'.format(key))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    output_path = os.path.join(config_train['output_path'], 'roc-all')

    plt.savefig(output_path + '.pdf')
    plt.savefig(output_path + '.png')



if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
