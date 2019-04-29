#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
from array import array
import yaml
import pickle
import numpy as np
import os

from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

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
    parser.add_argument("config_testing", help="Path to testing config file")
    parser.add_argument("fold", type=int, help="Trained model to be tested.")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load config %s.", filename)
    return yaml.load(open(filename, "r"))

def check_for_duplicates(data, current_model_dictionary):
    for i_model, model in enumerate(data["results"]):
        if model["number_of_variables"] == current_model_dictionary["number_of_variables"]:
            data["results"][i_model] = current_model_dictionary
            return data
    data["results"].append(current_model_dictionary)
    return data


def main(args, config_test, config_train):
    path = os.path.join(config_train["output_path"],
                        config_test["preprocessing"][args.fold])
    logger.debug("Load preprocessing %s.", path)
    preprocessing = pickle.load(open(path, "rb"))

    path = os.path.join(config_train["output_path"],
                        config_test["model"][args.fold])
    logger.debug("Load keras model %s.", path)
    model = load_model(path, compile=False)

    path = os.path.join(config_train["datasets"][(1, 0)[args.fold]])
    logger.debug("Loop over test dataset %s to get model response.", path)
    file_ = ROOT.TFile(path)
    classes = config_train["classes"]
    variables_used = config_train["variables"]
    all_responses = []
    all_true_labels = []
    all_weights = []
    class_weights = config_train["class_weights"]
    for i_class, class_ in enumerate(classes):
        logger.debug("Process class %s.", class_)

        tree = file_.Get(class_)
        if tree == None:
            logger.fatal("Tree %s does not exist.", class_)
            raise Exception

        values = []
        for variable in variables_used:
            typename = tree.GetLeaf(variable).GetTypeName()
            if typename == "Float_t":
                values.append(array("f", [-999]))
            elif typename == "Int_t":
                values.append(array("i", [-999]))
            else:
                logger.fatal("Variable {} has unknown type {}.".format(variable, typename))
                raise Exception
            tree.SetBranchAddress(variable, values[-1])

        weight = array("f", [-999])
        tree.SetBranchAddress(config_test["weight_branch"], weight)

        values_preprocessed = np.zeros(
            (tree.GetEntries(), len(values)), dtype=np.float32)
        y_class = np.zeros((tree.GetEntries(), len(classes)))
        y_class[:, i_class] = np.ones((tree.GetEntries()))
        weights = np.zeros(tree.GetEntries(), dtype=np.float32)
        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)
            values_stacked = np.hstack(values).reshape(1, len(values))
            values_preprocessed[i_event, :] = preprocessing.transform(
                values_stacked)
            weights[i_event] = weight[0]

        response = model.predict(values_preprocessed)
        response = np.squeeze(response)
        y_class = np.argmax(y_class, axis=-1)
        response = np.argmax(response, axis=-1)

        correct_predictions = np.sum(np.equal(y_class, response))

        logger.info("For class {} we have {} correct predictions of {} total events.".format(class_, correct_predictions, tree.GetEntries()))

        all_responses.append(response)
        all_true_labels.append(y_class)
        all_weights.append(weights*class_weights[class_])

    all_responses = np.concatenate(all_responses, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_weights = np.concatenate(all_weights, axis=0)

    recall = recall_score(all_true_labels, all_responses, average=None, sample_weight=all_weights)
    precision = precision_score(all_true_labels, all_responses, average=None, sample_weight=all_weights)
    f1 = f1_score(all_true_labels, all_responses, average=None, sample_weight=all_weights)
    logger.info("The precision score is {}, the recall is {} and the f1 score is {}".format(precision,
                                                                                            recall,
                                                                                            f1))
    class_dictionary = dict()
    class_dictionary["Precision"] = precision.tolist()
    class_dictionary["Recall"] = recall.tolist()
    class_dictionary["F1-Score"] = f1.tolist()

    model_dictionary = dict()
    model_dictionary["model"] = config_train["model"]
    model_dictionary["variables_used"] = variables_used
    model_dictionary["number_of_variables"] = len(variables_used)
    model_dictionary["class_weights"] = config_train["class_weights"]
    model_dictionary["scores_by_class"] = class_dictionary
    json_path = os.path.join(config_train["output_path_json"], "pruning_information_fold{}.json".format(args.fold))

    import json

    if not os.path.exists(json_path):
        if not os.path.exists(config_train["output_path_json"]):
            os.mkdir(config_train["output_path_json"])

        result_dict = {"results": []}
        with open(json_path, "w") as f:
            json.dump(result_dict, f)

    with open(json_path, 'r') as f:
        data = json.load(f)
    data = check_for_duplicates(data=data, current_model_dictionary=model_dictionary)
    with open(json_path, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    args = parse_arguments()
    config_test = parse_config(args.config_testing)
    config_train = parse_config(args.config_training)
    main(args, config_test, config_train)