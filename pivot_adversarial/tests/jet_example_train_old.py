import numpy as np
import pickle
import sys

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam

# Command line arguments
lam = float(sys.argv[1])
seed = int(sys.argv[2])
np.random.seed = seed

# Prepare data
fd = open("/home/sjoerger/workspace/pivot_adversarials/jet_example/jets-pile.pickle", "rb")
X_train, X_test, y_train, y_test, z_train, z_test, scaler = pickle.load(fd)
X = X_train
y = y_train
z = z_train
fd.close()

# mask = (z_train[:, 0] == 1)
# X_train = X_train[mask]
# y_train = y_train[mask]
# z_train = z_train[mask]

# Set network architectures
inputs = Input(shape=(X.shape[1],))
Dx = Dense(64, activation="tanh")(inputs)
Dx = Dense(64, activation="relu")(Dx)
Dx = Dense(64, activation="relu")(Dx)
Dx = Dense(1, activation="sigmoid")(Dx)
D = Model(input=[inputs], output=[Dx])

Rx = D(inputs)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(z.shape[1], activation="softmax")(Rx)
R = Model(input=[inputs], output=[Rx])


def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * K.binary_crossentropy(y_pred, y_true)

    return loss_D


def make_loss_R(c):
    def loss_R(z_true, z_pred):
        return c * K.categorical_crossentropy(z_pred, z_true)

    return loss_R


opt_D = Adam()
D.compile(loss=['binary_crossentropy'], optimizer=opt_D)

opt_DRf = SGD(momentum=0)
DRf = Model(input=[inputs], output=[D(inputs), R(inputs)])
DRf.compile(loss=[make_loss_D(c=1.0),
                  make_loss_R(c=-lam)],
            optimizer=opt_DRf)

opt_DfR = SGD(momentum=0)
DfR = Model(input=[inputs], output=[R(inputs)])
DfR.compile(loss=[make_loss_R(c=1.0)],
            optimizer=opt_DfR)

# Pretraining of D
D.trainable = True
R.trainable = False
D.fit(X_train, y_train, nb_epoch=20)

DfR.summary()

# Pretraining of R
if lam > 0.0:
    D.trainable = False
    R.trainable = True
    DfR.fit(X_train[y_train == 0], z_train[y_train == 0], nb_epoch=20)
    # DfR.fit(X_train, z_train, nb_epoch=20)

# Adversarial training
batch_size = 128

print(DRf.evaluate(X_test, [y_test, z_test]))

for i in range(1001):
    print(i)

    # Fit D
    D.trainable = True
    R.trainable = False
    DRf.compile()
    indices = np.random.permutation(len(X_train))[:batch_size]
    DRf.train_on_batch(X_train[indices], [y_train[indices], z_train[indices]])

    # Fit R
    if lam > 0.0:
        D.trainable = False
        R.trainable = True
        DRf.compile()
        DfR.fit(X_train[y_train == 0], z_train[y_train == 0],
                batch_size=batch_size, nb_epoch=1, verbose=0)
        # DfR.fit(X_train, z_train,
        #         batch_size=batch_size, nb_epoch=1, verbose=0)

D.save_weights("old-%.4f-%d.h5" % (lam, seed))
