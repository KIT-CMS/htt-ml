import keras.backend as K
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import SGD, Adam
import numpy as np
import os
import logging
logger = logging.getLogger("pivot_models")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def classifier(num_inputs, num_outputs, layers, dropout_rate=0.2):
    inputs = Input(shape=(num_inputs,))

    for i, nodes in enumerate(layers):
        if i == 0:
            layer = Dense(nodes, activation=None, name='Classifier_Dense_{}'.format(i))(inputs)
        else:
            layer = Dense(nodes, activation=None, name='Classifier_Dense_{}'.format(i))(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(rate=dropout_rate)(layer)

    output = Dense(num_outputs, activation=None, name='Classifier_output')(layer)
    #output = BatchNormalization()(output)
    output = Activation('sigmoid', name='Classifier_sigmoid')(output)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def classifier_catagorical(num_inputs, num_outputs, layers, dropout_rate=0.2):
    inputs = Input(shape=(num_inputs,))

    for i, nodes in enumerate(layers):
        if i == 0:
            layer = Dense(nodes, activation=None, name='Classifier_Dense_{}'.format(i))(inputs)
        else:
            layer = Dense(nodes, activation=None, name='Classifier_Dense_{}'.format(i))(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(rate=dropout_rate)(layer)

    output = Dense(num_outputs, activation=None, name='Classifier_output')(layer)
    #output = BatchNormalization()(output)
    output = Activation('softmax', name='Classifier_softmax')(output)

    model = Model(inputs=[inputs], outputs=[output])

    return model


def adversary_softmax(num_inputs, num_outputs, layers):
    inputs = Input(shape=(num_inputs,))

    for i, nodes in enumerate(layers):
        if i == 0:
            layer = Dense(nodes, activation=None, name='adversary_Dense_{}'.format(i))(inputs)
        else:
            layer = Dense(nodes, activation=None, name='adversary_Dense_{}'.format(i))(layer)
        layer = Activation('relu')(layer)

    output = Dense(num_outputs, activation=None, name='adversary_output')(layer)
    output = Activation('softmax', name='adversary_softmax')(output)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def adversary_regression(num_inputs, num_outputs, layers):
    inputs = Input(shape=(num_inputs,))

    for i, nodes in enumerate(layers):
        if i == 0:
            layer = Dense(nodes, activation=None, name='adversary_Dense_{}'.format(i))(inputs)
        else:
            layer = Dense(nodes, activation=None, name='adversary_Dense_{}'.format(i))(layer)
        layer = Activation('relu')(layer)

    output = Dense(num_outputs, activation=None, name='adversary_output')(layer)
    output = Activation('linear', name='adversary_linear')(output)

    model = Model(inputs=[inputs], outputs=[output])

    return model

class PivotAdversarialClassifier(object):

    def __init__(self, num_inputs, num_outputs, adv_outputs, lam):
        self.lam = lam

        clf_net = self._create_clf_net(num_inputs, num_outputs)
        adv_net = self._create_adv_net(num_outputs, adv_outputs)
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(num_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(num_inputs, clf_net, adv_net)
        self._val_metrics = None
        self._fairness_metrics = None

        self.predict = self._clf.predict

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag

        return make_trainable

    def summary(self):
        print('Classifier:')
        self._clf.summary()
        print('Adversary:')
        self._adv.summary()
        print('Combined net:')
        self._clf_w_adv.summary()

    def _create_clf_net(self, num_inputs, num_outputs):
        return classifier(num_inputs=num_inputs, num_outputs=num_outputs, layers=[64]*3)

    def _create_adv_net(self, num_inputs, num_ouputs):
        return adversary_softmax(num_inputs=num_inputs, num_outputs=num_ouputs, layers=[64]*3)

    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        optimizer = Adam()
        clf.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return clf

    def _compile_clf_w_adv(self, num_inputs, clf_net, adv_net):
        inputs = Input(shape=(num_inputs,))
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs),adv_net(clf_net(inputs))])
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.,-self.lam]
        optimizer = SGD()
        clf_w_adv.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                          loss_weights=loss_weights,
                          optimizer=optimizer)
        return clf_w_adv

    def _compile_adv(self, num_inputs, clf_net, adv_net):
        inputs = Input(shape=(num_inputs,))
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        optimizer = SGD()
        adv.compile(loss=['categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
        return adv

    def pretrain_classifier(self, x, y, epochs=10, verbose=0):
        self._trainable_clf_net(True)
        self._clf.fit(x, y, epochs=epochs, verbose=verbose)

    def pretrain_adversary(self, x,z, epochs=10, verbose=0):
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        self._adv.fit(x,z, epochs=epochs, verbose=verbose)

    def save(self, path, fold):
        classifier_path = os.path.join(path, 'fold{}_classifier-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=classifier_path, overwrite=True)
        adversary_path = os.path.join(path, 'fold{}_adversary-l={}.h5'.format(fold,self.lam))
        self._adv.save(filepath=adversary_path, overwrite=True)
        combined_path = os.path.join(path, 'fold{}_combined-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=combined_path, overwrite=True)

    def fit(self, x_class, x_adv, y_class, z_class, z_adv, validation_data=None, n_iter=200, batch_size=128, verbose=0):
        if validation_data is not None:
            x_val, y_val, z_val = validation_data

        losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

        for idx in range(n_iter):
            if idx % 10 == 0 and validation_data is not None:
                loss = self._clf_w_adv.evaluate(x_val, [y_val, z_val], verbose=verbose)
                losses["L_f - L_r"].append(loss[0])
                losses["L_f"].append(loss[1])
                losses["L_r"].append(loss[2])
                print('L_r: {}'.format(losses["L_r"][-1]))
                print('L_f: {}'.format(losses['L_f'][-1]))
                print('L_f - L_r: {}'.format(losses["L_f - L_r"][-1]))

            # train adverserial
            if self.lam > 0.0:
                self._trainable_clf_net(False)
                self._trainable_adv_net(True)
                self._adv.fit(x_adv, z_adv, epochs=1, verbose=verbose)

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x_class))[:batch_size]
            self._clf_w_adv.train_on_batch(x_class[indices], [y_class[indices],z_class[indices]])

        return losses

class ToyAdversarialClassifier(object):

    def __init__(self, num_inputs, num_outputs, adv_outputs, lam, dropout):
        self.lam = lam
        self.dropout = dropout

        clf_net = self._create_clf_net(num_inputs, num_outputs)
        if adv_outputs == 1:
            adv_net = self._create_regression_adversary(num_outputs, adv_outputs)
        else:
            adv_net = self._create_classification_adversary(num_outputs, adv_outputs)
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(num_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(num_inputs, clf_net, adv_net)
        self._val_metrics = None
        self._fairness_metrics = None

        self.predict = self._clf.predict

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag

        return make_trainable

    def summary(self):
        print('Classifier:')
        self._clf.summary()
        print('Adversary:')
        self._adv.summary()
        print('Combined net:')
        self._clf_w_adv.summary()

    def _create_clf_net(self, num_inputs, num_outputs):
        return classifier(num_inputs=num_inputs, num_outputs=num_outputs, layers=[64]*3, dropout_rate=self.dropout)

    def _create_regression_adversary(self, num_inputs, num_ouputs):
        return adversary_regression(num_inputs=num_inputs, num_outputs=num_ouputs, layers=[64]*3)

    def _create_classification_adversary(self, num_inputs, num_ouputs):
        return adversary_softmax(num_inputs=num_inputs, num_outputs=num_ouputs, layers=[64]*3)

    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        optimizer = Adam()
        clf.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return clf

    def _compile_clf_w_adv(self, num_inputs, clf_net, adv_net):
        inputs = Input(shape=(num_inputs,))
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs),adv_net(clf_net(inputs))])
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.,-self.lam]
        optimizer = SGD()
        clf_w_adv.compile(loss=['binary_crossentropy', 'mean_squared_error'],
                          loss_weights=loss_weights,
                          optimizer=optimizer)
        return clf_w_adv

    def _compile_adv(self, num_inputs, clf_net, adv_net):
        inputs = Input(shape=(num_inputs,))
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        optimizer = SGD()
        adv.compile(loss=['mean_squared_error'], optimizer=optimizer)
        return adv

    def pretrain_classifier(self, x, y, path, fold, epochs=10, verbose=0, ):
        self._trainable_clf_net(True)
        self._clf.fit(x, y, epochs=epochs, verbose=verbose)
        classifier_path = os.path.join(path, 'fold{}_pre_classifier-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=classifier_path, overwrite=True)
        logger.info("Saved normal classifier to {}".format(classifier_path))

    def pretrain_adversary(self, x,z, epochs=10, verbose=0):
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        self._adv.fit(x,z, epochs=epochs, verbose=verbose)

    def save(self, path, fold):
        classifier_path = os.path.join(path, 'fold{}_classifier-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=classifier_path, overwrite=True)
        adversary_path = os.path.join(path, 'fold{}_adversary-l={}.h5'.format(fold,self.lam))
        self._adv.save(filepath=adversary_path, overwrite=True)
        combined_path = os.path.join(path, 'fold{}_combined-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=combined_path, overwrite=True)
        logger.info("Saved all models to {}".format(path))

    def fit(self, x_class, x_adv, y_class, z_class, z_adv, validation_data=None, n_iter=200, batch_size=128):
        if validation_data is not None:
            x_val, y_val, z_val = validation_data

        losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

        for idx in range(n_iter):
            if idx % 10 == 0 and validation_data is not None:
                print('Currently on batch {}'.format(idx))
                loss = self._clf_w_adv.evaluate(x_val, [y_val, z_val], verbose=0)
                losses["L_f - L_r"].append(loss[0])
                losses["L_f"].append(loss[1])
                losses["L_r"].append(loss[2])
                print('L_r: {}'.format(losses["L_r"][-1]))
                print('L_f: {}'.format(losses['L_f'][-1]))
                print('L_f - L_r: {}'.format(losses["L_f - L_r"][-1]))

            # train adverserial
            if self.lam > 0.0:
                self._trainable_clf_net(False)
                self._trainable_adv_net(True)
                self._adv.fit(x_adv, z_adv, epochs=1, verbose=0)

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x_class))[:batch_size]
            self._clf_w_adv.train_on_batch(x_class[indices], [y_class[indices],z_class[indices]])

        return losses

class RealAdversarialClassifier(object):

    def __init__(self, num_inputs, num_outputs, adv_outputs, lam, dropout):
        self.lam = lam
        self.dropout = dropout

        clf_net = self._create_clf_net(num_inputs, num_outputs)
        if adv_outputs == 1:
            self._adv_net = self._create_regression_adversary(num_outputs, adv_outputs)
        else:
            self._adv_net = self._create_classification_adversary(num_outputs, adv_outputs)
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(self._adv_net)
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(num_inputs, clf_net, self._adv_net)
        self._adv = self._compile_adv(num_inputs, clf_net, self._adv_net)
        self._val_metrics = None
        self._fairness_metrics = None

        self.predict = self._clf.predict

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag

        return make_trainable

    def summary(self):
        print('Classifier:')
        self._clf.summary()
        print('Standalone adversary:')
        self._adv_net.summary()
        print('Adversary:')
        self._adv.summary()
        print('Combined net:')
        self._clf_w_adv.summary()

    def _create_clf_net(self, num_inputs, num_outputs):
        return classifier_catagorical(num_inputs=num_inputs, num_outputs=num_outputs, layers=[200]*2, dropout_rate=self.dropout)

    def _create_regression_adversary(self, num_inputs, num_ouputs):
        return adversary_regression(num_inputs=num_inputs, num_outputs=num_ouputs, layers=[200]*2)

    def _create_classification_adversary(self, num_inputs, num_ouputs):
        logger.info('Creating classification adversary')
        return adversary_softmax(num_inputs=num_inputs, num_outputs=num_ouputs, layers=[64]*2)

    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        optimizer = Adam()
        clf.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return clf

    def _compile_clf_w_adv(self, num_inputs, clf_net, adv_net):
        inputs = Input(shape=(num_inputs,))
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs),adv_net(clf_net(inputs))])
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.,-self.lam]
        optimizer = Adam()
        clf_w_adv.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                          loss_weights=loss_weights,
                          optimizer=optimizer)
        return clf_w_adv

    def _compile_adv(self, num_inputs, clf_net, adv_net):
        inputs = Input(shape=(num_inputs,))
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        optimizer = Adam()
        adv.compile(loss=['categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
        return adv

    def pretrain_classifier(self, x, y, sample_weights, batch_size, path, fold, epochs=10, verbose=0, ):
        self._trainable_clf_net(True)
        self._clf.fit(x, y, sample_weight=sample_weights, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
        classifier_path = os.path.join(path, 'fold{}_pre_classifier-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=classifier_path, overwrite=True)
        logger.info("Saved normal classifier to {}".format(classifier_path))

    def pretrain_adversary(self, x,z, class_weights=None, batch_size=124, epochs=10, verbose=0):
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        self._adv.fit(x,z, epochs=epochs, class_weight=class_weights, batch_size=batch_size, verbose=verbose, shuffle=True)

    def evaluate_adversary(self, x,z):
        return self._adv.evaluate(x,z)

    def save(self, path, fold):
        classifier_path = os.path.join(path, 'fold{}_classifier-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=classifier_path, overwrite=True)
        adversary_path = os.path.join(path, 'fold{}_adversary-l={}.h5'.format(fold,self.lam))
        self._adv.save(filepath=adversary_path, overwrite=True)
        combined_path = os.path.join(path, 'fold{}_combined-l={}.h5'.format(fold,self.lam))
        self._clf.save(filepath=combined_path, overwrite=True)
        logger.info("Saved all models to {}".format(path))

    def fit(self, x_class, x_adv, y_class, z_class, z_adv, sample_weights, class_weights_adv=None, validation_data=None, n_iter=200, batch_size=128, verbose=0):
        if validation_data is not None:
            x_val, y_val, z_val, w_val = validation_data

        losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

        w_val_adv = np.ones((len(w_val)))
        w_val_adv[(z_val[:,0] == 1)] = 2.

        for idx in range(n_iter):
            if idx % 10 == 0 and validation_data is not None:
                print('Currently on batch {}'.format(idx))
                loss = self._clf_w_adv.evaluate(x_val, [y_val, z_val], sample_weight=[w_val, w_val_adv], verbose=verbose)
                losses["L_f - L_r"].append(loss[0])
                losses["L_f"].append(loss[1])
                losses["L_r"].append(loss[2])
                print('L_r: {}'.format(losses["L_r"][-1]))
                print('L_f: {}'.format(losses['L_f'][-1]))
                print('L_f - L_r: {}'.format(losses["L_f - L_r"][-1]))

            # train adverserial
            if self.lam > 0.0:
                self._trainable_clf_net(False)
                self._trainable_adv_net(True)
                self._adv.fit(x_adv, z_adv, class_weight=class_weights_adv, epochs=1, verbose=verbose)

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x_class))[:batch_size]
            self._clf_w_adv.train_on_batch(x_class[indices], [y_class[indices],z_class[indices]], sample_weight=[sample_weights[0][indices], sample_weights[1][indices]])

        return losses