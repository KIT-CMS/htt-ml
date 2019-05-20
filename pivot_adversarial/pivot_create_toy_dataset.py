import numpy as np
import yaml
import pickle
import os
np.random.seed(333)
import argparse

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
    adv_output = config['adv_output']
    if adv_output == "binned":
        bins = config["bins"]
        if not bins:
            raise Exception('Please set number of bins when choosing binned as adv_output.')

    logger.info('Creating {} toy examples with {} adversarial output.'.format(n_samples, adv_output))

    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples // 2)
    z = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

    X1[:, 1] += z[n_samples // 2:]

    X = np.vstack([X0, X1])
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1

    if adv_output == "binned":
        Z = np.zeros((n_samples, bins))
        cuts = np.linspace(-2.0, 2.0, bins)
        for i, c_i in enumerate(cuts):
            for j, value in enumerate(z):
                if i == 0:
                    if value <= c_i:
                        Z[j, i] = 1
                else:
                    if value <= c_i and value > cuts[i - 1]:
                        Z[j, i] = 1
    else:
        Z = z

    X_train, X_valid, y_train, y_valid, z_train, z_valid = train_test_split(X, y, Z, test_size=1.-config["train_test_split"], random_state=333)

    data_path = os.path.join(output_path, 'toy_example_n={}_{}.pickle'.format(n_samples, adv_output))

    with open(data_path, 'wb') as file:
        pickle.dump([X_train, X_valid, y_train, y_valid, z_train, z_valid], file)

    logger.info('Sucessfully pickled the training data.')

if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)