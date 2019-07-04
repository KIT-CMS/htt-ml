import argparse
import pickle
import yaml
import json
import copy

from keras.models import load_model
import keras2sequential


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export model for Htt analyses to .json format")
    parser.add_argument("config_training", help="Path to training config file")
    parser.add_argument("config_application", help="Path to application config file")
    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"))


def main(args, config_training, config_application):
    # Create template dictionary for variables.json
    variables_template = {
        "class_labels" : config_training["classes"],
        "inputs" : [{"name" : v, "offset" : 0.0, "scale" : 1.0} for v in config_training["variables"]]
    } 
    
    # Load keras model and preprocessing
    for c, p, w, v, a in zip(config_application["classifiers"],
                             config_application["preprocessing"],
                             config_application["weights"],
                             config_application["variable_exports"],
                             config_application["architecture_exports"]):
        # export weights in .h5 format & model in .json format
        model = load_model(c, compile=False)
        model_type = model.__class__.__name__
        if model_type != "Sequential":
            model = keras2sequential.model2sequential(model)
            assert model.__class__.__name__ == 'Sequential'
            save_path = c.split('.')[0]+'_sequential.h5'
            model.save(filepath=save_path)
            print('Changed model type to sequential and saved it to {}'.format(save_path))
        architecture = model.to_json()
        model.save_weights(w, overwrite=True)
        with open(a, "w") as f:
            f.write(architecture)
            f.close()
        # export scale & offsets vor variables
        variables = copy.deepcopy(variables_template)
        scaler = pickle.load(open(p, "rb"))
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
