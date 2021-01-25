import argparse
import pickle
import yaml
import json
import copy

from tensorflow.keras.models import load_model

import logging
logger = logging.getLogger("Export keras to json")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export model for Htt analyses to .json format")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_application", help="Path to application config file")
    parser.add_argument("--conditional", required=False, type=bool, default=False, help="Use one network for all eras or separate networks.")
    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"), Loader=yaml.SafeLoader)


def main(args, config_training, config_application):
    # Create template dictionary for variables.json
    logger.info("Use conditional network: {}".format(args.conditional))
    if args.conditional:
        eras = ["2016", "2017", "2018"]
    else:
        eras = []
    variables_template = {
        "class_labels" : config_training["classes"],
        "inputs" : [{"name" : v, "offset" : 0.0, "scale" : 1.0} for v in config_training["variables"] + eras]
    } 
    mldir=config_training["output_path"]+"/"
    # Load keras model and preprocessing
    for c, p, w, v, a in zip(config_application["classifiers"],
                             config_application["preprocessing"],
                             config_application["weights"],
                             config_application["variable_exports"],
                             config_application["architecture_exports"]):
        # export weights in .h5 format & model in .json format
        c=mldir+c
        p=mldir+p
        w=mldir+w
        v=mldir+v
        a=mldir+a
        model = load_model(c)
        model.save_weights(w)
        with open(a, "w") as f:
            f.write(model.to_json())
            f.close()
        # export scale & offsets vor variables
        variables = copy.deepcopy(variables_template)
        scaler = pickle.load(open(p, "rb"), encoding="bytes")
        for variable,offset,scale in zip(variables["inputs"],scaler.mean_,scaler.scale_): # NOTE: offsets & scales are in the same order as in config_training["variables"]
            variable["offset"] = -offset
            variable["scale"] = 1.0/scale
        with open(v, "w") as f:
            f.write(json.dumps(variables))
            f.close()


if __name__ == "__main__":
    args = parse_arguments()
    config_application = parse_config(args.config_application)
    config_training = parse_config(args.config_training)
    main(args, config_training, config_application)
