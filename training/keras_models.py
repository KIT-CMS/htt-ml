from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
from keras_custom_metrics import *
from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def example(num_inputs, num_outputs):
    """
    Example Keras model
    """
    model = Sequential()
    model.add(
        Dense(
            10, init="glorot_normal", activation="relu", input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_uniform", activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=[
            "categorical_accuracy",
        ])
    return model

def smhtt_dropout_relu(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([200] * 2):
        if i == 0:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5), input_dim=num_inputs))
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5)))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))

    model.add(Dense(num_outputs, kernel_regularizer=l2(1e-5)))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), weighted_metrics=["mean_squared_error"], metrics=['accuracy'])
    return model

def smhtt_significance_multi_output(num_inputs, num_outputs, output_names, learning_rate = 1e-3):
    inputs = Input(shape=(num_inputs,))
    weights = Input(shape=(1,))

    layer_1 = Dense(200, activation=None, kernel_regularizer=None)(inputs)
    layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    #layer_1 = Dropout(rate=0.3)(layer_1)
    layer_2 = Dense(200, activation=None, kernel_regularizer=None)(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
    #layer_2 = Dropout(rate=0.3)(layer_2)
    output_list = []
    for i, name in enumerate(output_names):
        if i == 0:
            output_0 = Dense(num_outputs, activation=None, kernel_regularizer=None, name="dense_{}".format(name))(layer_2)
            output_0 = BatchNormalization(name='batchnorm_{}'.format(name))(output_0)
            output_0 = Activation('softmax', name=name)(output_0)
            output_list.append(output_0)
            continue
        new_output = Lambda(lambda x: x, name=name)(output_list[0])
        output_list.append(new_output)

    model = Model(inputs=[inputs, weights], outputs=output_list)

    loss_dict = dict()
    for i, name in enumerate(output_names):
        loss_dict[name] = wrapped_partial(significance_loss_multi_output(class_label=i), weights=weights)

    model.compile(loss=loss_dict, optimizer=Adam(lr=learning_rate), loss_weights=None)
    return model

def smhtt_significance_unbinned(num_inputs, num_outputs, output_names, learning_rate = 1e-3):
    inputs = Input(shape=(num_inputs,))
    weights = Input(shape=(1,))

    layer_1 = Dense(200, activation=None, kernel_regularizer=l2(1e-5))(inputs)
    #layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    #layer_1 = Dropout(rate=0.3)(layer_1)
    layer_2 = Dense(200, activation=None, kernel_regularizer=l2(1e-5))(layer_1)
    #layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
    #layer_2 = Dropout(rate=0.3)(layer_2)
    output_0 = Dense(num_outputs, activation=None, kernel_regularizer=l2(1e-5), name="dense_out")(layer_2)
    #output_0 = BatchNormalization(name='batchnorm_out')(output_0)
    output_0 = Activation('softmax', name='out')(output_0)

    model = Model(inputs=[inputs, weights], outputs=output_0)

    metrics = []
    loss_ce_ = wrapped_partial(loss_ce, weights=weights)
    metrics.append(loss_ce_)
    for i, name in enumerate(output_names):
        variable_loss = wrapped_partial(significance_unbinned_class, event_weights=weights, class_label=i)
        metrics.append(variable_loss)

    #loss = wrapped_partial(significance_loss_unbinned(number_of_labels=num_outputs), weights=weights)
    loss = wrapped_partial(loss_ce, weights=weights)

    model.compile(loss=loss, optimizer=Adam(lr=1e-4), metrics=metrics)

    return model

def smhtt_significance_binned(num_inputs, num_outputs, output_names, learning_rate = 1e-3):
    inputs = Input(shape=(num_inputs,))
    weights = Input(shape=(1,))

    layer_1 = Dense(200, activation=None, kernel_regularizer=l2(1e-5))(inputs)
    #layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    #layer_1 = Dropout(rate=0.3)(layer_1)
    layer_2 = Dense(200, activation=None, kernel_regularizer=l2(1e-5))(layer_1)
    #layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
    #layer_2 = Dropout(rate=0.3)(layer_2)
    output_0 = Dense(num_outputs, activation=None, kernel_regularizer=l2(1e-5), name="dense_out")(layer_2)
    #output_0 = BatchNormalization(name='batchnorm_out')(output_0)
    output_0 = Activation('softmax', name='out')(output_0)

    model = Model(inputs=[inputs, weights], outputs=output_0)

    metrics = []
    loss_ce_ = wrapped_partial(loss_ce, weights=weights)
    metrics.append(loss_ce_)
    for i, name in enumerate(output_names):
        variable_loss = wrapped_partial(significance_loss_binned, event_weights=weights, class_label=i)
        metrics.append(variable_loss)

    loss = wrapped_partial(significance_loss_binned(number_of_labels=num_outputs), weights=weights)

    model.compile(loss=loss, optimizer=Adam(lr=1e-4), metrics=metrics)

    return model

def smhtt_ams_multi_output(num_inputs, num_outputs, output_names, learning_rate = 1e-3):
    inputs = Input(shape=(num_inputs,))
    weights = Input(shape=(1,))

    layer_1 = Dense(200, activation=None, kernel_regularizer=None)(inputs)
    layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    #layer_1 = Dropout(rate=0.3)(layer_1)
    layer_2 = Dense(200, activation=None, kernel_regularizer=None)(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
    #layer_2 = Dropout(rate=0.3)(layer_2)
    output_list = []
    for i, name in enumerate(output_names):
        if i == 0:
            output_0 = Dense(num_outputs, activation=None, kernel_regularizer=None, name="dense_{}".format(name))(layer_2)
            output_0 = BatchNormalization(name='batchnorm_{}'.format(name))(output_0)
            output_0 = Activation('softmax', name=name)(output_0)
            output_list.append(output_0)
            continue
        new_output = Lambda(lambda x: x, name=name)(output_list[0])
        output_list.append(new_output)

    model = Model(inputs=[inputs, weights], outputs=output_list)

    loss_dict = dict()
    for i, name in enumerate(output_names):
        loss_dict[name] = wrapped_partial(ams_loss_multi_output(class_label=i), weights=weights)

    model.compile(loss=loss_dict, optimizer=Adam(lr=learning_rate), loss_weights=None)
    return model

def smhtt_ams_single_output(num_inputs, num_outputs, output_names, learning_rate = 1e-3):
    inputs = Input(shape=(num_inputs,))
    weights = Input(shape=(1,))

    layer_1 = Dense(200, activation=None, kernel_regularizer=None)(inputs)
    #layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    #layer_1 = Dropout(rate=0.3)(layer_1)
    layer_2 = Dense(200, activation=None, kernel_regularizer=None)(layer_1)
    #layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
    #layer_2 = Dropout(rate=0.3)(layer_2)
    output_0 = Dense(num_outputs, activation=None, kernel_regularizer=None, name="dense_out")(layer_2)
    #output_0 = BatchNormalization(name='batchnorm_out')(output_0)
    output_0 = Activation('softmax', name='out')(output_0)

    model = Model(inputs=[inputs, weights], outputs=output_0)

    metrics = []
    loss_ce_ = wrapped_partial(loss_ce, w=weights)
    metrics.append(loss_ce_)
    for i, name in enumerate(output_names):
        variable_loss = wrapped_partial(ams_loss_class, event_weights=weights, class_label=i, br=1.0)
        metrics.append(variable_loss)

    loss = wrapped_partial(ams_loss_single_output(number_of_labels=num_outputs), weights=weights)

    model.compile(loss=loss, optimizer=Adam(lr=learning_rate), metrics=metrics)
    return model


def smhtt_dropout_tanh(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([200] * 2):
        if i == 0:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5), input_dim=num_inputs))
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5)))
        model.add(Activation("tanh"))
        model.add(Dropout(0.3))

    model.add(Dense(num_outputs, kernel_regularizer=l2(1e-5)))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), weighted_metrics=["mean_squared_error"], metrics=['accuracy'])
    return model


def smhtt_dropout_tanh_tensorflow(input_placeholder, keras_model):
    weights = {}
    for layer in keras_model.layers:
        print("Layer: {}".format(layer.name))
        for weight, array in zip(layer.weights, layer.get_weights()):
            print("    weight, shape: {}, {}".format(weight.name,
                                                     np.array(array).shape))
            weights[weight.name] = np.array(array)

    w1 = tf.get_variable('w1', initializer=weights['dense_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['dense_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['dense_2/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['dense_2/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['dense_3/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['dense_3/bias:0'])

    l1 = tf.tanh(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.tanh(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return l3


def smhtt_dropout_tensorflow(input_placeholder, keras_model):
    weights = {}
    for layer in keras_model.layers:
        print("Layer: {}".format(layer.name))
        for weight, array in zip(layer.weights, layer.get_weights()):
            print("    weight, shape: {}, {}".format(weight.name,
                                                     np.array(array).shape))
            weights[weight.name] = np.array(array)

    w1 = tf.get_variable('w1', initializer=weights['dense_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['dense_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['dense_2/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['dense_2/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['dense_3/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['dense_3/bias:0'])

    l1 = tf.nn.relu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.relu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return l3
