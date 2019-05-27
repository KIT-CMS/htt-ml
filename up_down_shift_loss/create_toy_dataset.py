import numpy as np
import yaml
import pickle
import os
np.random.seed(333)
import argparse
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger("create_toy_dataset")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(description="Create training dataset")
    parser.add_argument("config", help="config_train for output paths")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load YAML config: {}".format(filename))
    return yaml.load(open(filename, "r"))

def main(args, config):
    n_samples = config['n_samples']
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    uncertainty_mean = config['uncertainty_mean']

    logger.info('Creating {} toy examples.'.format(n_samples*2))

    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples)
    w_0 = np.ones(n_samples)
    w_1 = np.random.normal(loc=1.0, scale=uncertainty_mean, size=n_samples)
    w_1 = np.clip(w_1, 1e-5, 2.0)
    #w_1 = np.add(w_1,abs(np.min(w_1)))

    X = np.vstack([X0, X1])
    w = np.concatenate((w_0, w_1))
    y = np.zeros(n_samples*2)
    y[n_samples:] = 1

    plt.title("$X$")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="b", marker="o", edgecolors="none", label=r'$y=1$')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="r", marker="o", edgecolors="none", label=r'$y=0$')
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig(output_path + '/toy_distribution.png')
    plt.clf()

    plt.title("$W$")
    plt.hist(w_1, histtype='step', normed=True, lw=3, label='Weights', bins=50)
    plt.legend()
    plt.savefig(output_path + '/weight_distribution.png')
    plt.clf()

    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(X, y, w, test_size=1.-config["train_test_split"], random_state=333)

    data_path = os.path.join(output_path, 'toy_example_n={}_mu={}.pickle'.format(n_samples, uncertainty_mean))

    with open(data_path, 'wb') as file:
        pickle.dump([X_train, X_valid, y_train, y_valid, w_train, w_valid], file)

    logger.info('Sucessfully pickled the training data.')

if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)