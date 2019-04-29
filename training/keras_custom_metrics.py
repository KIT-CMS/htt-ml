from sklearn.metrics import recall_score, precision_score, f1_score
from keras import backend as K
from keras.callbacks import Callback
from keras.losses import categorical_crossentropy
import tensorflow as tf
import numpy as np

class Recall_Precision(Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val, sample_weights = self.validation_data[0], self.validation_data[1], self.validation_data[2]
        y_predict = np.asarray(self.model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        recall = recall_score(y_val, y_predict, average=None, sample_weight=sample_weights)
        precision = precision_score(y_val, y_predict, average=None, sample_weight=sample_weights)
        f1 = f1_score(y_val, y_predict, average=None, sample_weight=sample_weights)
        print("F1 score: {} - Recall: {} - Precision: {}".format(f1, recall, precision))

        self._data.append({
            'F1-Score': f1,
            'Recall': recall,
            'Precision': precision
        })
        return

    def get_data(self):
        return self._data

def calculate_significance_per_class(y_true, y_pred, event_weights, number_of_labels):
    class_dictionary = {}
    for i in range(number_of_labels):
        signal = 0
        background = 0

        histograms = [signal, background]

        class_dictionary[str(i)] = histograms

    for y_true_instance, y_pred_instance, event_weight in zip(y_true,y_pred, event_weights):
        histograms = class_dictionary[str(np.argmax(y_pred_instance))]
        if np.argmax(y_true_instance) == np.argmax(y_pred_instance):
            histograms[0] += event_weight

        else:
            histograms[1] += event_weight

    class_signifiance= dict()
    for i in range(number_of_labels):
        histograms = class_dictionary[str(i)]
        signal_histogram = histograms[0]
        background_histogram = histograms[1]
        significance = signal_histogram / np.sqrt(signal_histogram + background_histogram + 0.000001)
        class_signifiance[str(i)] = significance

    return class_signifiance

def calculate_significance_per_bin(y_true, y_pred, event_weights, number_of_labels, nbins):
    class_dictionary = {}
    for i in range(number_of_labels):
        signal = []
        background = []
        for j in range(nbins):
            signal.append(0.)
            background.append(0.)

        histograms = [signal, background]

        class_dictionary[str(i)] = histograms

    for y_true_instance, y_pred_instance, event_weight in zip(y_true,y_pred, event_weights):
        histograms = class_dictionary[str(np.argmax(y_pred_instance))]
        max_value = np.max(y_pred_instance)
        if np.argmax(y_true_instance) == np.argmax(y_pred_instance):
            signal_histogram = histograms[0]
            for i in range(len(signal_histogram)):
                if max_value < 1./float(nbins)*(float(i)+1.):
                    signal_histogram[i] += event_weight
                    break

        else:
            background_histogram = histograms[1]
            for i in range(len(background_histogram)):
                if max_value < 1./float(nbins)*(float(i)+1.):
                    background_histogram[i] += event_weight
                    break

    class_significance = []
    for i in range(number_of_labels):
        histograms = class_dictionary[str(i)]
        signal_histogram = np.array(histograms[0], dtype=np.float32)
        background_histogram = np.array(histograms[1], dtype=np.float32)
        significance_per_bin = signal_histogram / np.sqrt(signal_histogram + background_histogram + 0.000001)
        class_significance.append(significance_per_bin)

    return class_significance


class significance(Callback):
    def __init__(self, x_train, y_train, w_train):
        Callback.__init__(self)
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, event_weights, y_val = self.validation_data[0], self.validation_data[1], self.validation_data[2]
        y_predict = np.asarray(self.model.predict([X_val, event_weights]))
        #y_predict_train = np.asarray(self.model.predict([self.x_train, self.w_train]))

        class_significance = calculate_significance_per_class(y_true= y_val, y_pred= y_predict[0], event_weights=event_weights, number_of_labels=np.shape(y_val)[1])
        #class_significance_train = calculate_significance_per_class(y_true= self.y_train, y_pred= y_predict_train[0], event_weights=self.w_train, number_of_labels=np.shape(self.y_train)[1])

        print(" - Significance: {}".format(class_significance))
        #print(" - Train-Significance: {}".format(class_significance_train))

        self._data.append({
            'Significance': class_significance
        })
        return

    def get_data(self):
        return self._data


def ams_loss(y_true, y_pred, event_weights, class_label, br=0.):
    highest_values = K.max(y_pred, axis=-1)

    label_0_mask = K.cast(K.equal(K.argmax(y_pred), class_label), K.floatx())
    label_0_mask_true = K.cast(K.equal(K.argmax(y_true), class_label), K.floatx())
    event_weights = K.flatten(event_weights)

    signals = label_0_mask*label_0_mask_true*highest_values*event_weights
    all = label_0_mask*highest_values*event_weights

    signal_sum = K.sum(signals)

    all_sum = K.sum(all)

    background_sum = all_sum - signal_sum

    ams_score = K.sqrt(K.clip(2*((all_sum + br)*K.log(1+signal_sum/(background_sum + br)) - signal_sum), min_value=K.epsilon(), max_value=1e10))

    ams_score_negative = - ams_score

    return ams_score_negative

def significance_loss(y_true, y_pred, event_weights, class_label):
    highest_values = K.max(y_pred, axis=-1)

    label_mask = K.cast(K.equal(K.argmax(y_pred), class_label), K.floatx())
    label_mask_true = K.cast(K.equal(K.argmax(y_true), class_label), K.floatx())
    event_weights = K.flatten(event_weights)

    signals = label_mask*label_mask_true*highest_values*event_weights
    all_events = label_mask*highest_values*event_weights


    signal_sum = K.sum(signals)


    all_sum = K.sum(all_events)
    significance_positive = (signal_sum) / (all_sum + K.epsilon())

    significance_negative = - significance_positive

    return significance_negative

def significance_curry_loss_2(number_of_labels):
    def significance(y_true, y_pred, weights):
        total_loss = 0
        for i in range(number_of_labels):
            loss = significance_loss(y_true, y_pred, weights, i)
            total_loss += loss
        return total_loss

    return significance

def significance_curry_loss(class_label):
    def significance(y_true, y_pred, weights):
        class_loss = significance_loss(y_true, y_pred, event_weights=weights, class_label=class_label)
        return class_loss

    return significance

def ams_curry_loss(class_label, br = 1.0):
    def ams(y_true, y_pred, weights):
        class_loss = ams_loss(y_true, y_pred, event_weights=weights, class_label=class_label, br = br)
        return class_loss

    return ams


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def ggh_precision(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)

    # get a tensor that is 1 when class is ggh and 0 if not.
    true_indece = K.argmax(y_true)
    ggh_indice = K.cast(K.equal(true_indece, 0), K.floatx())

    # Multiply this tensor with the prediction to only get predictions of ggh.
    tp = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())*ggh_indice)
    all_positive_predictions = K.sum(K.cast(K.equal(K.argmax(y_pred),0), K.floatx())) + K.epsilon()

    precision = tp / all_positive_predictions
    return precision

def ggh_recall(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)

    # get a tensor that is 1 when class is ggh and 0 if not.
    true_indece = K.argmax(y_true)
    ggh_indice = K.cast(K.equal(true_indece, 0), K.floatx())

    tp = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())*ggh_indice)
    all_possible_positives = K.sum(ggh_indice) + K.epsilon()

    recall = tp / all_possible_positives
    return recall

def qqh_precision(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)

    # get a tensor that is 1 when class is qqh and 0 if not.
    true_indece = K.argmax(y_true)
    qqh_indice = K.cast(K.equal(true_indece, 1), K.floatx())

    # Multiply this tensor with the prediction to only get predictions of qqh.
    tp = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())*qqh_indice)
    all_positive_predictions = K.sum(K.cast(K.equal(K.argmax(y_pred),1), K.floatx())) + K.epsilon()

    precision = tp / all_positive_predictions

    #precision = K.print_tensor(precision, message='Precision before masking: ')

    return precision

def qqh_recall(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)

    # get a tensor that is 1 when class is qqh and 0 if not.
    true_indece = K.argmax(y_true)
    qqh_indice = K.cast(K.equal(true_indece, 1), K.floatx())

    tp = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())*qqh_indice)
    all_possible_positives = K.sum(qqh_indice) + K.epsilon()

    recall = tp / all_possible_positives
    return recall


def ggh_fbeta(y_true, y_pred):
    beta = 1

    precision = ggh_precision(y_true, y_pred)
    recall = ggh_recall(y_true, y_pred)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

def qqh_fbeta(y_true, y_pred):
    beta = 1

    precision = qqh_precision(y_true, y_pred)
    recall = qqh_recall(y_true, y_pred)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
