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
    parser.add_argument('--config_training', default=[], nargs='+')
    parser.add_argument("--config_application", help="Path to application config file")
    parser.add_argument("--conditional", required=False, type=bool, default=False, help="Use one network for all eras or separate networks.")
    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"), Loader=yaml.SafeLoader)


def main(args, config_application):
    # Create template dictionary for variables.json
    logger.info("Use conditional network: {}".format(args.conditional))
    if args.conditional:
        eras = ["2016", "2017", "2018"]
    else:
        eras = [] 
    
    
    variables_template = {"input_sequences":[],
                        "inputs": [],
                        "outputs": []
                        }
    for i,config_file in enumerate(args.config_training):
        config = parse_config(config_file)
        heavy_mass = config["output_path"].split("/")[-1].split("_")[-2]
        light_mass = config["output_path"].split("/")[-1].split("_")[-1]
        variables_template["inputs"].append({"name": "node_{hm}_{lm}".format(hm=heavy_mass, lm=light_mass),                    
                                            "variables" : [{"name" : v, "offset" : 0.0, "scale" : 1.0} for v in config["variables"] + eras]
                                            })
        variables_template["outputs"].append({"labels": config["classes"],
                                                "name": "output_{hm}_{lm}".format(hm=heavy_mass, lm=light_mass)
                                                })
    train_path = config["training_path"]
    # Load keras model and preprocessing
    p=0
    for c, w, v, a in zip(config_application["classifiers"],
                             config_application["weights"],
                             config_application["variable_exports"],
                             config_application["architecture_exports"]):
        # export weights in .h5 format & model in .json format
        print(p)
        c=train_path+c
        w=train_path+w
        v=train_path+v
        a=train_path+a
        model = load_model(c)
        model.save_weights(w)
        with open(a, "w") as f:
            f.write(model.to_json())
            f.close()
        # export scale & offsets vor variables
        variables = copy.deepcopy(variables_template)
        
        for i,config_file in enumerate(args.config_training):            
            config = parse_config(config_file)
            output_path = config["output_path"]+"/"
            preprocess_files = config_application["preprocessing"]
            scaler = pickle.load(open(output_path+preprocess_files[p], "rb"), encoding="bytes")
            for variable,offset,scale in zip(variables["inputs"][i]["variables"],scaler.mean_,scaler.scale_): 
                # NOTE: offsets & scales are in the same order as in config_training["variables"]
                variable["offset"] = -offset
                variable["scale"] = 1.0/scale
        print(variables)
        with open(v, "w") as f:
            f.write(json.dumps(variables))
            f.close()
        p+=1


if __name__ == "__main__":
    args = parse_arguments()
    config_application = parse_config(args.config_application)
    main(args, config_application)
