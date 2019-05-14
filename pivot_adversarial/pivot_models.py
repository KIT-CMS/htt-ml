import keras.backend as K
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import SGD, Adam
import numpy as np
import os

def classifier(num_inputs, num_outputs, layers):
    inputs = Input(shape=(num_inputs,))

    for i, nodes in enumerate(layers):
        if i == 0:
            layer = Dense(nodes, activation=None, name='Classifier_Dense_{}'.format(i))(inputs)
        else:
            layer = Dense(nodes, activation=None, name='Classifier_Dense_{}'.format(i))(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(rate=0.2)(layer)

    output = Dense(num_outputs, activation=None, name='Classifier_output')(layer)
    #output = BatchNormalization()(output)
    output = Activation('sigmoid', name='Classifier_sigmoid')(output)

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

    def fit(self, x_class, x_adv, y_class, z_class, z_adv, validation_data=None, n_iter=200, batch_size=128):
        if validation_data is not None:
            x_val, y_val, z_val = validation_data

        losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

        for idx in range(n_iter):
            if idx % 10 == 0 and validation_data is not None:
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
                self._adv.fit(x_adv, z_adv, batch_size=batch_size, epochs=1, verbose=0)

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x_class))[:batch_size]
            self._clf_w_adv.train_on_batch(x_class[indices], [y_class[indices],z_class[indices]])

        return losses