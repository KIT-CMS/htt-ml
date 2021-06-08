#!/usr/bin/env python

import logging

logger = logging.getLogger("keras_training")

import argparse
import yaml
import os
import pickle


def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(
        description="Train machine Keras models for Htt analyses")
    parser.add_argument("config", help="Path to training config file")
    parser.add_argument("fold", type=int, help="Select the fold to be trained")
    parser.add_argument(
        "--conditional",
        required=False,
        type=bool,
        default=False,
        help="Use one network for all eras or separate networks.")
    parser.add_argument(
        "--randomization",
        required=False,
        type=bool,
        default=False,
        help=
        "Randomize signal classes for conditional training in case one era has insufficient signal data."
    )
    parser.add_argument(
        "--balance-batches",
        required=False,
        type=bool,
        default=False,
        help=
        "Use a equal amount of events of each class in a batch and normalize those by dividing each individual event weigth by the sum of event weight of the respective class in that batch "
    )
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Parse config.")
    return yaml.load(open(filename, "r"), Loader=yaml.SafeLoader)


def setup_logging(level, output_file=None):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not output_file == None:
        file_handler = logging.FileHandler(output_file, "w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def main(args, config):
    logger.info(args)
    # Set seed and import packages
    # NOTE: This need to be done before any keras module is imported!
    logger.debug("Import packages and set random seed to %s.",
                 int(config["seed"]))
    import numpy as np
    np.random.seed(int(config["seed"]))

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
    import root_numpy

    # NOTE: Matplotlib needs to be imported after Keras/TensorFlow because of conflicting libraries #TODO: Works now for LCG94?
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    import tensorflow as tf
    logger.debug(tf.__file__)
    # check tf version and  GPU availability
    print("Using {} from {}".format(tf.__version__, tf.__file__))
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("Default GPU Devices: {}".format(physical_devices))
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    else:
        print("No GPU found. Using CPU.")

    tf.compat.v1.set_random_seed(int(config["seed"]))
    from datetime import datetime
    # dir for tensorboard
    log_dir = "logs/" + str(datetime.now().strftime("%H_%M_%S"))
    from sklearn import preprocessing, model_selection
    import keras_models
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

    # Extract list of variables
    variables = config["variables"]
    classes = config["classes"]
    logger.debug("Use variables:")
    for v in variables:
        logger.debug("%s", v)

    if args.randomization:
        signal_classes = [
            class_ for class_ in classes
            if class_.startswith(('ggh', 'qqh', 'vbftopo'))
        ]
        randomization_era = "2016"
    else:
        signal_classes = []
        randomization_era = None

    # Perform data transformations on gpu if available
    if physical_devices:
        device = '/GPU:0'
    else:
        device = '/CPU'
    with tf.device(device):
        # Load training dataset
        if args.conditional:
            args.balanced_batches = True
            eras = ['2016', '2017', '2018']
            len_eras = len(eras)
        else:
            eras = ['any']
            len_eras = 0
        x = []  # Training input
        y = []  # Target classes
        w = []  # Weights for training
        z = []  # Era information for batching
        for i_era, era in enumerate(eras):
            if args.conditional:
                filename = config["datasets_{}".format(era)][args.fold]
            else:
                filename = config["datasets"][args.fold]
            logger.debug("Load training dataset from {}.".format(filename))
            rfile = ROOT.TFile(filename, "READ")
            x_era = []
            y_era = []
            w_era = []
            for i_class, class_ in enumerate(classes):
                logger.debug("Process class %s.", class_)
                tree = rfile.Get(class_)
                if tree == None:
                    logger.fatal("Tree %s not found in file %s.", class_,
                                 filename)
                    raise Exception
                friend_trees_names = [
                    k.GetName() for k in rfile.GetListOfKeys()
                    if k.GetName().startswith("_".join([class_, "friend"]))
                ]
                for friend in friend_trees_names:
                    tree.AddFriend(friend)

                # Get inputs for this class

                x_class = np.zeros(
                    (tree.GetEntries(), len(variables) + len_eras))
                x_conv = root_numpy.tree2array(tree, branches=variables)
                for i_var, var in enumerate(variables):
                    x_class[:, i_var] = x_conv[var]
                if np.any(np.isnan(x_class)):
                    logger.fatal(
                        "Nan in class {} for era {} in file {} for any of {}".
                        format(class_, era, filename, variables))
                    raise Exception

                # One hot encode eras if conditional. Additionally randomize signal class eras if desired.
                if args.conditional:
                    if (class_ in signal_classes) and args.randomization:
                        logger.debug("Randomizing class {}".format(class_))
                        random_era = np.zeros((tree.GetEntries(), len_eras))
                        for event in random_era:
                            idx = np.random.randint(3, size=1)
                            event[idx] = 1
                        x_class[:, -3:] = random_era
                    else:
                        if era == "2016":
                            x_class[:, -3] = np.ones((tree.GetEntries()))
                        elif era == "2017":
                            x_class[:, -2] = np.ones((tree.GetEntries()))
                        elif era == "2018":
                            x_class[:, -1] = np.ones((tree.GetEntries()))
                x_era.append(x_class)

                # Get weights
                w_class = np.zeros((tree.GetEntries(), 1))
                w_conv = root_numpy.tree2array(
                    tree, branches=[config["event_weights"]])
                if args.balance_batches:
                    w_class[:, 0] = w_conv[config["event_weights"]]
                else:
                    if args.conditional:
                        w_class[:,
                                0] = w_conv[config["event_weights"]] * config[
                                    "class_weights_{}".format(era)][class_]
                    else:
                        w_class[:, 0] = w_conv[config[
                            "event_weights"]] * config["class_weights"][class_]
                if np.any(np.isnan(w_class)):
                    logger.fatal(
                        "Nan in weight class {} for era {} in file {}.".format(
                            class_, era, filename))
                    raise Exception
                w_era.append(w_class)

                # Get targets for this class
                y_class = np.zeros((tree.GetEntries(), len(classes)))
                y_class[:, i_class] = np.ones((tree.GetEntries()))
                y_era.append(y_class)

            # Stack inputs, targets and weights to a Keras-readable dataset
            x_era = np.vstack(x_era)  # inputs
            y_era = np.vstack(y_era)  # targets
            w_era = np.vstack(w_era)  # weights
            z_era = np.zeros((y_era.shape[0], len(eras)))  # era information
            z_era[:, i_era] = np.ones((y_era.shape[0]))
            x.append(x_era)
            y.append(y_era)
            w.append(w_era)
            z.append(z_era)

        # Stack inputs, targets and weights to a Keras-readable dataset
        x = np.vstack(x)  # inputs
        y = np.vstack(y)  # targets
        w = np.vstack(w)  # weights
        w = np.squeeze(w)  # needed to get weights into keras
        z = np.vstack(z)  # era information

        # Perform input variable transformation and pickle scaler object.
        # Only perform transformation on continuous variables
        x_scaler = x[:, :len(variables)]
        logger.info("Use preprocessing method %s.", config["preprocessing"])
        if "standard_scaler" in config["preprocessing"]:
            scaler = preprocessing.StandardScaler().fit(x_scaler)
            for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
                logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                             var, mean, std)
        elif "identity" in config["preprocessing"]:
            scaler = preprocessing.StandardScaler().fit(x_scaler)
            for i in range(len(scaler.mean_)):
                scaler.mean_[i] = 0.0
                scaler.scale_[i] = 1.0
            for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
                logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                             var, mean, std)
        elif "robust_scaler" in config["preprocessing"]:
            scaler = preprocessing.RobustScaler().fit(x_scaler)
            for var, mean, std in zip(variables, scaler.center_,
                                      scaler.scale_):
                logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                             var, mean, std)
        elif "min_max_scaler" in config["preprocessing"]:
            scaler = preprocessing.MinMaxScaler(
                feature_range=(-1.0, 1.0)).fit(x_scaler)
            for var, min_, max_ in zip(variables, scaler.data_min_,
                                       scaler.data_max_):
                logger.debug("Preprocessing (variable, min, max): %s, %s, %s",
                             var, min_, max_)
        elif "quantile_transformer" in config["preprocessing"]:
            scaler = preprocessing.QuantileTransformer(
                output_distribution="normal",
                random_state=int(config["seed"])).fit(x_scaler)
        else:
            logger.fatal("Preprocessing %s is not implemented.",
                         config["preprocessing"])
            raise Exception
        x[:, :len(variables)] = scaler.transform(x_scaler)

        path_preprocessing = os.path.join(
            config["output_path"],
            "fold{}_keras_preprocessing.pickle".format(args.fold))
        logger.info("Write preprocessing object to %s.", path_preprocessing)
        pickle.dump(scaler, open(path_preprocessing, 'wb'))

        # Split data in training and testing
        x_train, x_test, y_train, y_test, w_train, w_test, z_train, z_test = model_selection.train_test_split(
            x,
            y,
            w,
            z,
            test_size=1.0 - config["train_test_split"],
            random_state=int(config["seed"]))
        del x, y, w
    # Add callbacks
    callbacks = []
    if "early_stopping" in config["model"]:
        logger.info("Stop early after %s tries.",
                    config["model"]["early_stopping"])
        callbacks.append(
            EarlyStopping(patience=config["model"]["early_stopping"]))

    logger.info("Saving tensorboard logs in {}".format(log_dir))
    callbacks.append(
        TensorBoard(log_dir=log_dir,
                    profile_batch="2, {}".format(
                        config["model"]["steps_per_epoch"])))

    path_model = os.path.join(config["output_path"],
                              "fold{}_keras_model.h5".format(args.fold))
    if os.path.exists(path_model):
        print("Path {} already exists! I will not overwrite it".format(
            path_model))
        raise Exception
    if "save_best_only" in config["model"]:
        if config["model"]["save_best_only"]:
            logger.info("Write best model to %s.", path_model)
            callbacks.append(
                ModelCheckpoint(path_model, save_best_only=True, verbose=1))

    if "reduce_lr_on_plateau" in config["model"]:
        logger.info("Reduce learning-rate after %s tries.",
                    config["model"]["reduce_lr_on_plateau"])
        callbacks.append(
            ReduceLROnPlateau(patience=config["model"]["reduce_lr_on_plateau"],
                              verbose=1))

    # Train model
    if not hasattr(keras_models, config["model"]["name"]):
        logger.fatal("Model %s is not implemented.", config["model"]["name"])
        raise Exception
    logger.info("Train keras model %s.", config["model"]["name"])

    if config["model"]["eventsPerClassAndBatch"] < 0:
        eventsPerClassAndBatch = x_train.shape[0] * len(classes)
    else:
        eventsPerClassAndBatch = config["model"]["eventsPerClassAndBatch"]

    ###
    classIndexDict = {
        label: np.where(y_train[:, i_class] == 1)[0]
        for i_class, label in enumerate(classes)
    }
    if "steps_per_epoch" in config["model"]:
        steps_per_epoch = int(config["model"]["steps_per_epoch"]) * len(eras)
        recommend_steps_per_epoch = int(
            min([len(classIndexDict[class_])
                 for class_ in classes]) / (eventsPerClassAndBatch)) + 1
        logger.info(
            "steps_per_epoch: Using {} instead of recommended minimum of {}".
            format(str(steps_per_epoch), str(recommend_steps_per_epoch)))
    else:
        logger.info("model: steps_per_epoch: Not found in {} ".format(
            args.config))
        raise Exception

    model_impl = getattr(keras_models, config["model"]["name"])
    model = model_impl(len(variables) + len_eras, len(classes))
    model.summary()
    if (args.balance_batches):
        logger.info("Running on balanced batches.")
        # Loop over all eras and classes to divide the batch equally, defines the indices of the corrects answers for each era/class combination
        eraIndexDict = {
            era: {
                label: np.where((z_train[:, i_era] == 1)
                                & (y_train[:, i_class] == 1))[0]
                for i_class, label in enumerate(classes)
            }
            for i_era, era in enumerate(eras)
        }
        import threading

        # Sequence to generate data batches
        class balancedBatchGenerator:
            def __init__(self):
                self.eventsPerClassAndBatch = eventsPerClassAndBatch
                self.lock = threading.Lock()

            def __iter__(self):
                return self

            def __next__(self):
                with self.lock:
                    nperClass = int(eventsPerClassAndBatch)
                    # Choose eventsPerClassAndBatch events randomly for each class and era
                    selIdxDict = {
                        era: {
                            label: eraIndexDict[era][label][np.random.randint(
                                0, len(eraIndexDict[era][label]), nperClass)]
                            for label in classes
                        }
                        for era in eras
                    }
                    y_collect = np.concatenate([
                        y_train[selIdxDict[era][label]] for label in classes
                        for era in eras
                    ])
                    x_collect = np.concatenate([
                        x_train[selIdxDict[era][label], :] for label in classes
                        for era in eras
                    ])
                    w_collect = np.concatenate([
                        w_train[selIdxDict[era][label]] *
                        (eventsPerClassAndBatch /
                         np.sum(w_train[selIdxDict[era][label]]))
                        for label in classes for era in eras
                    ])
                    # return list of choosen values
                    return x_collect, y_collect, w_collect

        def calculateValidationWeights(x_test, y_test, w_test):
            testIndexDict = {
                era: {
                    label: np.where((z_test[:, i_era] == 1)
                                    & (y_test[:, i_class] == 1))[0]
                    for i_class, label in enumerate(classes)
                }
                for i_era, era in enumerate(eras)
            }

            y_collect = np.concatenate([
                y_test[testIndexDict[era][label]] for label in classes
                for era in eras
            ])
            x_collect = np.concatenate([
                x_test[testIndexDict[era][label], :] for label in classes
                for era in eras
            ])
            w_collect = np.concatenate([
                w_test[testIndexDict[era][class_]] *
                (len(x_test) / np.sum(w_test[testIndexDict[era][class_]]))
                for class_ in classes for era in eras
            ])

            return x_collect, y_collect, w_collect

        x_test, y_test, w_test = calculateValidationWeights(
            x_test, y_test, w_test)

        # define tf.data.Dataset for input generator data
        traindata = tf.data.Dataset.from_generator(
            balancedBatchGenerator,
            output_signature=(tf.TensorSpec(
                shape=(eventsPerClassAndBatch * len(classes) *
                       max(1, len(eras)), len(variables) + len_eras),
                dtype=tf.float64),
                              tf.TensorSpec(
                                  shape=(eventsPerClassAndBatch *
                                         len(classes) * max(1, len(eras)),
                                         len(classes)),
                                  dtype=tf.float64),
                              tf.TensorSpec(
                                  shape=(eventsPerClassAndBatch *
                                         len(classes) * max(1, len(eras))),
                                  dtype=tf.float64)))
        # define tf.data.Dataset for validation data
        validata = tf.data.Dataset.from_tensor_slices((x_test, y_test, w_test))
        # collect all data into single batch
        validata = validata.batch(len(x_test))
        # Prefetch datasets to CPU or GPU if available
        if physical_devices:
            traindata = traindata.apply(
                tf.data.experimental.prefetch_to_device(
                    tf.test.gpu_device_name(), tf.data.experimental.AUTOTUNE))
            validata = validata.apply(
                tf.data.experimental.prefetch_to_device(
                    tf.test.gpu_device_name(), tf.data.experimental.AUTOTUNE))
        else:
            traindata = traindata.prefetch(tf.data.experimental.AUTOTUNE)
            validata = validata.prefetch(tf.data.experimental.AUTOTUNE)

        print("Timestamp training start: {}".format(
            datetime.now().strftime("%H:%M:%S")))
        # Train model with prefetched datasets
        history = model.fit(traindata,
                            steps_per_epoch=steps_per_epoch,
                            epochs=config["model"]["epochs"],
                            callbacks=callbacks,
                            validation_data=validata,
                            verbose=2)

    else:
        with tf.device(device):
            traindata = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train, w_train))
            traindata = traindata.batch(eventsPerClassAndBatch * len(classes))
            validata = tf.data.Dataset.from_tensor_slices(
                (x_test, y_test, w_test))
            validata = validata.batch(len(x_test))
            if physical_devices:
                traindata = traindata.apply(
                    tf.data.experimental.prefetch_to_device(
                        tf.test.gpu_device_name(),
                        tf.data.experimental.AUTOTUNE))
                validata = validata.apply(
                    tf.data.experimental.prefetch_to_device(
                        tf.test.gpu_device_name(),
                        tf.data.experimental.AUTOTUNE))
            else:
                traindata = traindata.prefetch(tf.data.experimental.AUTOTUNE)
                validata = validata.prefetch(tf.data.experimental.AUTOTUNE)

        from datetime import datetime
        logger.info("Timestamp training start: {}".format(
            datetime.now().strftime("%H:%M:%S")))
        history = model.fit(traindata,
                            validation_data=validata,
                            epochs=config["model"]["epochs"],
                            shuffle=True,
                            callbacks=callbacks,
                            verbose=2)

    logger.info("Timestamp training end: {}".format(
        datetime.now().strftime("%H:%M:%S")))
    # Plot loss
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
    plt.plot(epochs,
             history.history["val_loss"],
             lw=3,
             label="Validation loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    path_plot = os.path.join(config["output_path"],
                             "fold{}_loss".format(args.fold))
    plt.legend()
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")

    # Save model
    if not "save_best_only" in config["model"]:
        logger.info("Write model to %s.", path_model)
        model.save(path_model)


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config)
    setup_logging(logging.INFO,
                  "{}/training{}.log".format(config["output_path"], args.fold))
    main(args, config)
