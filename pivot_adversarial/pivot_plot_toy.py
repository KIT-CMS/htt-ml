#!/usr/bin/env python
import yaml
import pickle
import numpy as np
import argparse
import os
import glob
import matplotlib as mpl
import time

mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve

import logging
logger = logging.getLogger("pivot_plot_toy")
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

def make_X(n_samples, z):
    np.random.seed(int(time.time()))
    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples // 2)
    X1[:, 1] += z
    X = np.vstack([X0, X1])
    y0 = np.zeros(n_samples //2)
    y1 = np.ones(n_samples //2)

    return X0, X1

def prepare_data(x0, x1, discriminator):
    pred = discriminator.predict(x0)
    pred_signal = [i for i in pred if i <= 0.5]
    pred_background_0 = [i for i in pred if i > 0.5]

    pred = discriminator.predict(x1)
    pred_signal_1 = [i for i in pred if i > 0.5]
    pred_background = [i for i in pred if i <= 0.5]

    return pred_signal, pred_background

def make_shifted_examples(n_samples, z):
    np.random.seed(int(time.time()))
    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples // 2)
    X1[:, 1] += z
    X = np.vstack([X0, X1])
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1

    return X, y, z

def plot_roc(tpr_plain, fpr_plain, auc_plain, tpr_adv, fpr_adv, auc_adv, title):

    plt.figure(figsize=(10, 8))

    for key, item in fpr_plain.items():
        fpr_lam = item
        tpr_lam = tpr_plain[key]
        AUC = auc_plain[key]
        plt.plot(fpr_lam, tpr_lam, lw=3, label='Plain: Shift={}, AUC={}'.format(key, AUC))
    for key, item in fpr_adv.items():
        fpr_lam = item
        tpr_lam = tpr_adv[key]
        AUC = auc_adv[key]
        plt.plot(fpr_lam, tpr_lam, lw=3, label='Adversary: Shift={}, AUC={}'.format(key, AUC))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    output_path = title

    plt.savefig(output_path + '.pdf')
    plt.savefig(output_path + '.png')
    logger.info("Saved roc curve to {}".format(title))
    pass

def plot_histograms(discriminator, title):
    X1,X0 = make_X(200000, z = -1)
    X = np.vstack([X1,X0])
    pred_down = discriminator.predict(X)
    pred_down = np.reshape(pred_down, np.shape(pred_down)[0])
    X1,X0 = make_X(200000, z = 0)
    X = np.vstack([X1,X0])
    pred_nom = discriminator.predict(X)
    pred_nom = np.reshape(pred_nom, np.shape(pred_nom)[0])
    X1,X0 = make_X(200000, z = +1)
    X = np.vstack([X1,X0])
    pred_up = discriminator.predict(X)
    pred_up = np.reshape(pred_up, np.shape(pred_up)[0])
    plt.hist(pred_down, bins=50, normed=1, lw=3, histtype="step", label="$p(f(X)|Z=-\sigma)$")
    plt.hist(pred_nom, bins=50, normed=1, lw=3, histtype="step", label="$p(f(X)|Z=0)$")
    plt.hist(pred_up, bins=50, normed=1, lw=3, histtype="step", label="$p(f(X)|Z=+\sigma)$")
    plt.legend(loc="best")
    plt.xlabel("$f(X)$")
    plt.ylabel("$p(f(X))$")
    plt.grid()
    plt.savefig(title + ".png")
    plt.savefig(title + ".pdf")
    logger.info("Saved normal histogram plot to {}".format(title))
    plt.clf()
    pass

def plot_decision_surface(discriminator, title):
    plt.figure(figsize=(6,6))
    from matplotlib.mlab import griddata
    X_test = np.random.rand(30000, 2)
    X_test[:, 0] *= 10.
    X_test[:, 0] -= 5.
    X_test[:, 1] *= 10.
    X_test[:, 1] -= 5.

    y_pred = discriminator.predict(X_test)

    y_pred = np.reshape(y_pred, np.shape(y_pred)[0])

    xi = np.linspace(-1., 2., 100)
    yi = np.linspace(-1., 3, 100)
    zi = griddata(x=X_test[:, 0], y=X_test[:, 1], z=y_pred, xi=xi, yi=yi, interp="linear")
    CS = plt.contourf(xi, yi, zi, 20, cmap=plt.cm.viridis,
                      vmax=1.0, vmin=0.0)
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m.set_array(y_pred)
    m.set_clim(0., 1.)
    plt.colorbar(m, boundaries=np.linspace(0., 1., 10))
    # plt.colorbar()
    plt.scatter([0], [0], c="red", linewidths=0, label=r"$\mu_0$")
    plt.scatter([1], [0], c="blue", linewidths=0, label=r"$\mu_1|Z=z$")
    plt.scatter([1], [0 + 1], c="blue", linewidths=0)
    plt.scatter([1], [0 + 2], c="blue", linewidths=0)
    plt.text(1.2, 0 - 0.1, "$Z=-\sigma$", color="k")
    plt.text(1.2, 1 - 0.1, "$Z=0$", color="k")
    plt.text(1.2, 2 - 0.1, "$Z=+\sigma$", color="k")
    plt.xlim(-1, 2)
    plt.ylim(-1, 3)
    plt.legend(loc="upper left", scatterpoints=1)
    plt.savefig(title + ".png")
    plt.savefig(title + ".pdf")
    logger.info("Saved decision surface plot to {}".format(title))
    plt.clf()
    pass


def plot_significance(discriminator, title):
    nbins = 50

    fig, (ax, ax_2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))

    X0_nom, X1_nom = make_X(n_samples=400000, z=0)
    pred_0_nom = discriminator.predict(X0_nom)
    pred_1_nom = discriminator.predict(X1_nom)

    #print(pred_1_nom.ravel(), pred_0_nom.ravel())

    values_nominal, _, _ = ax.hist([pred_0_nom.ravel(), pred_1_nom.ravel()], bins=nbins, stacked=True, normed=0, histtype="step",
                                label=[r"$p(f(S))$", r"$p(f(B)|Z=0)$"], linewidth=3.0)

    X0_down, X1_down = make_X(n_samples=400000, z=-1)
    pred_1_down = discriminator.predict(X1_down)
    combined_down = np.vstack([pred_0_nom, pred_1_down])

    values_down, _, _ = ax.hist(combined_down, bins=nbins, normed=0, histtype="step",
                                label=r"$p(f(B)|Z=-\sigma)$", linewidth=3.0)


    X0_up, X1_up = make_X(n_samples=400000, z=+1)
    pred_1_up = discriminator.predict(X1_up)
    combined_up = np.vstack([pred_0_nom, pred_1_up])
    values_up, _, _ = ax.hist(combined_up, bins=nbins, normed=0,
                                        histtype="step",
                                        label= r"$p(f(B)|Z=+\sigma)$",
                                        linewidth=3.0)

    ax.legend(loc="best")
    ax.set_xlim([0., 1.0])
    ax.set_ylim([0.,60000.])
    plt.xlabel("$f(X)$")
    ax.set_ylabel("$Number of events$")
    plt.grid()

    background_values = values_nominal[1] - values_nominal[0]

    sigma_b = []
    significance = []
    x_values = []
    for i, bin in enumerate(values_down):
        sigma_b.append(np.abs((values_up[i] - bin) / 2.))

    for i, bin in enumerate(values_nominal[0]):
        signal_events = bin
        background_events = background_values[i]
        significance.append(signal_events / np.sqrt(signal_events + background_events + sigma_b[i]))
        x_values.append(float(i) / float(nbins) + 1. / float(nbins) / 2.)

    ax_2.plot(x_values, significance, 'ro')
    ax_2.set_xlim([0., 1.0])
    ax_2.set_ylabel(r'$S/\sqrt{S+B + \sigma_B}$')
    ax_2.set_ylim([0.,100.])
    fig.tight_layout()
    fig.savefig(title + ".png", bbox_inches="tight")
    fig.savefig(title + ".pdf", bbox_inches="tight")
    logger.info("Saved significance plot to {}".format(title))
    fig.clf()
    pass

def main(args, config_test, config_train):
    lam = config_train['model']['lambda']

    discriminator_path = os.path.join(config_train['output_path'], config_test['pre_classifier_path'].format(lam))

    discriminator = load_model(discriminator_path)

    adv_output = config_train['adv_output']

    title = os.path.join(config_train['output_path'], 'pre_training_{}'.format(adv_output))

    plot_significance(discriminator, title=title)

    plt.clf()

    title = os.path.join(config_train['output_path'], 'plain_hist_{}'.format(adv_output))

    plot_histograms(discriminator, title)

    plt.clf()

    title = os.path.join(config_train['output_path'], 'plain_surface_{}'.format(adv_output))

    plot_decision_surface(discriminator, title)

    plt.clf()

    test_shifts = [1,0,-1]

    tpr_plain = dict()
    fpr_plain = dict()
    auc_plain = dict()

    for shift in test_shifts:
        X_roc, y_roc, z_roc = make_shifted_examples(n_samples=200000, z=shift)
        fpr_plain[shift], tpr_plain[shift], _ = roc_curve(y_true=y_roc, y_score=discriminator.predict(X_roc))
        auc_plain[shift] = roc_auc_score(y_true=y_roc, y_score=discriminator.predict(X_roc))

    plt.clf()

    discriminator_path = os.path.join(config_train['output_path'], config_test['classifier_path'].format(lam))

    discriminator = load_model(discriminator_path)

    title = os.path.join(config_train['output_path'], 'pivot_{}'.format(adv_output))

    plot_significance(discriminator, title=title)

    plt.clf()

    title = os.path.join(config_train['output_path'], 'pivot_hist_{}'.format(adv_output))

    plot_histograms(discriminator, title)

    plt.clf()

    title = os.path.join(config_train['output_path'], 'pivot_surface_{}'.format(adv_output))

    plot_decision_surface(discriminator, title)

    plt.clf()

    tpr_adversary = dict()
    fpr_adversary = dict()
    auc_adversary = dict()


    for shift in test_shifts:
        X_roc, y_roc, z_roc = make_shifted_examples(n_samples=200000, z=shift)
        fpr_adversary[shift], tpr_adversary[shift], _ = roc_curve(y_true=y_roc, y_score=discriminator.predict(X_roc))
        auc_adversary[shift] = roc_auc_score(y_true=y_roc, y_score=discriminator.predict(X_roc))

    title = os.path.join(config_train['output_path'], 'roc_{}'.format(adv_output))

    plot_roc(tpr_plain,fpr_plain, auc_plain, tpr_adversary, fpr_adversary, auc_adversary, title)

if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)
