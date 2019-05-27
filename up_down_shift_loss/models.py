from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.activations import relu
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
import numpy as np
from functools import partial, update_wrapper

def threshold_relu(x, max=None, threshold=0.):
    x = relu(x, max_value=max)
    if max:
        x = x * K.cast(K.less(x, max), K.floatx())
    if threshold != 0:
        x = x * K.cast(K.greater(x, threshold), K.floatx())
    return x

def dummy_loss(y_true, y_pred):
    return (y_pred - y_pred)

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def up_down_loss(y_true, y_pred, event_weights, nbins):
    #highest_values = K.max(y_pred, axis=-1)
    #label_mask = K.cast(K.equal(K.argmax(y_pred), class_label), K.floatx())
    steps = []
    steps.append(0.0)
    event_weights = K.flatten(event_weights)
    bin_mask = threshold_relu(y_pred, max=1./nbins, threshold=0.0)
    nominal = K.sum(bin_mask/y_pred)/500.

    shifted = K.sum(bin_mask * event_weights/y_pred)/500.

    MSE = K.square((nominal - shifted))

    for step in np.linspace(1./nbins,1,nbins):
        bin_mask = threshold_relu(y_pred, max=step, threshold=steps[-1])
        #bin_mask = K.print_tensor(bin_mask, message='bin mask')
        nominal = K.sum(bin_mask/y_pred)/500.
        #event_weights = K.print_tensor(event_weights, message='weights')
        #nominal = K.print_tensor(nominal, message='nominal up')
        shifted = K.sum(bin_mask/y_pred*event_weights)/500.
        #shifted = K.print_tensor(shifted, message='shifted up')

        MSE += K.square((nominal - shifted))

        steps.append(step)

    #MSE = K.print_tensor(MSE, message='MSE')
    total_loss = MSE/nbins

    return total_loss


def curry_loss(nbins):
    def ams(y_true, y_pred, weights):
        class_loss = up_down_loss(y_true, y_pred, event_weights=weights,nbins=nbins)
        return class_loss

    return ams

def curry_loss_2(nbins):
    def ams(y_true, y_pred, weights):
        class_loss = loss_shift(y_true, y_pred, weights=weights, nbins=nbins)
        return class_loss

    return ams

def loss_ce(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def loss_ce_categorical(y_true, y_pred, weights):
    ce = K.mean(
        K.categorical_crossentropy(y_true, y_pred),
        axis=-1)
    w = K.flatten(weights)
    score_array = ce * w
    score_array /= K.mean(K.cast(K.not_equal(w, 0), K.floatx()))

    return score_array


def loss_shift(_, y_pred, weights, nbins):
    def gauss_filter(x, mean, width):
        return tf.exp(-1.0 * (x - mean)**2 / 2.0 / width**2)

    def count_bin(mean, width, weights):
        f = gauss_filter(y_pred, mean, width)
        return K.sum(f), K.sum(f * weights)

    bins = np.linspace(0, 1, nbins + 1)
    width = bins[1] - bins[0]
    mids = bins[:-1] + 0.5 * width
    l = 0.0
    for mean in mids:
        nominal, shifted = count_bin(mean, width, weights)
        l += K.square((nominal - shifted) / K.clip(nominal, K.epsilon(), None))
    l /= K.cast(len(mids), K.floatx())
    scale = 10.0
    return scale * l

def loss_shift_categorical(_, y_pred, weights, event_weights, nbins, class_label):
    def gauss_filter(x, mean, width):
        return tf.exp(-1.0 * (x - mean)**2 / 2.0 / width**2)

    def count_bin(values, mean, width, mask):
        f = gauss_filter(values, mean, width)*event_weights * mask
        return K.sum(f), K.sum(f * weights)

    label_mask = K.cast(K.equal(K.argmax(y_pred),class_label),K.floatx())
    label_mask = K.reshape(label_mask, K.shape(weights))
    highest_values = K.max(y_pred, keepdims=True)
    bins = np.linspace(0.0, 1.0, nbins + 1)
    width = bins[1] - bins[0]
    mids = bins[:-1] + 0.5 * width
    l = 0.0
    for mean in mids:
        nominal, shifted = count_bin(highest_values, mean, width, label_mask)
        #nominal = K.print_tensor(nominal, message='nominal {}'.format(mean))
        #shifted = K.print_tensor(shifted, message='shifted {}'.format(mean))
        l += K.square((nominal - shifted))
    l /= K.cast(len(mids), K.floatx())
    scale = .1
    return scale * l

def loss_categorical(y_true, y_pred, weights, up, down, label, lam=1., nbins = 10):
    lce = loss_ce_categorical(y_true, y_pred, weights)
    lup = loss_shift_categorical(None, y_pred, up, weights, nbins, label)
    ldown = loss_shift_categorical(None, y_pred, down, weights, nbins, label)
    return lce + lam*lup + lam*ldown

def loss_up_categorical(_, y_pred, up, weights, label, nbins=10):
    return loss_shift_categorical(_, y_pred, up, weights, nbins,label)

def loss_down_categorical(_, y_pred, down, weights, label, nbins=10):
    return loss_shift_categorical(_, y_pred, down, weights, nbins,label)

def loss(y_true, y_pred, weights, up, down, lam=1., nbins = 10):
    lce = loss_categorical(y_true, y_pred, weights)
    lup = loss_shift(None, y_pred, up, nbins)
    ldown = loss_shift(None, y_pred, down, nbins)
    return lce + lam*lup + lam*ldown


def loss_up(_, y_pred, weights, nbins=10):
    return loss_shift(_, y_pred, weights, nbins)

def loss_down(_, y_pred, weights, nbins=10):
    return loss_shift(_, y_pred, weights, nbins)

def up_down_model(num_inputs, num_outputs):
    inputs = Input(shape=(num_inputs,))
    weights_up = Input(shape=(1,))
    weights_down = Input(shape=(1,))
    hidden = Dense(200, activation="relu")(inputs)
    hidden = Dense(200, activation="relu")(hidden)
    f = Dense(num_outputs, activation="sigmoid")(hidden)
    model = Model(inputs=[inputs, weights_up, weights_down], outputs=f)

    loss_ = wrapped_partial(loss, up=weights_up, down=weights_down)
    loss_up_ = wrapped_partial(loss_up, weights=weights_up)
    loss_down_ = wrapped_partial(loss_down, weights=weights_down)

    # Compile model
    model.compile(
        optimizer="adam", loss=loss_, metrics=[loss_ce, loss_up_, loss_down_])
    model.summary()

    return model

def real_model(num_inputs, num_outputs, bins=10.):
    inputs = Input(shape=(num_inputs,))
    normal_weights = Input(shape=(1,))
    weights_up = Input(shape=(1,))
    weights_down = Input(shape=(1,))
    hidden = Dense(200, activation="relu")(inputs)
    hidden = Dense(200, activation="relu")(hidden)
    f = Dense(num_outputs, activation="softmax")(hidden)
    model = Model(inputs=[inputs, normal_weights, weights_up, weights_down], outputs=f)

    loss_ = wrapped_partial(loss_categorical, label=0,weights=normal_weights, up=weights_up, down=weights_down, nbins=bins)
    loss_ce_ = wrapped_partial(loss_ce_categorical, weights=normal_weights)
    loss_up_ = wrapped_partial(loss_up_categorical, up=weights_up, weights=normal_weights, label=0, nbins=bins)
    loss_down_ = wrapped_partial(loss_down_categorical, down=weights_down, weights=normal_weights, label=0, nbins=bins)

    # Compile model
    model.compile(
        optimizer="adam", loss=loss_, metrics=[loss_ce_, loss_up_, loss_down_])
    model.summary()

    return model

def normal_model(num_inputs, num_outputs, bins=10.):
    inputs = Input(shape=(num_inputs,))
    normal_weights = Input(shape=(1,))
    weights_up = Input(shape=(1,))
    weights_down = Input(shape=(1,))
    hidden = Dense(200, activation="relu")(inputs)
    hidden = Dense(200, activation="relu")(hidden)
    f = Dense(num_outputs, activation="softmax")(hidden)
    model = Model(inputs=[inputs, normal_weights, weights_up, weights_down], outputs=f)

    loss_ = wrapped_partial(loss_categorical, label=0,weights=normal_weights, up=weights_up, down=weights_down, nbins=bins)
    loss_ce_ = wrapped_partial(loss_ce_categorical, weights=normal_weights)
    loss_up_ = wrapped_partial(loss_up_categorical, up=weights_up, weights=normal_weights, label=0, nbins=bins)
    loss_down_ = wrapped_partial(loss_down_categorical, down=weights_down, weights=normal_weights, label=0, nbins=bins)

    # Compile model
    model.compile(
        optimizer="adam", loss=loss_ce_, metrics=[loss_ce_, loss_up_, loss_down_])
    model.summary()

    return model