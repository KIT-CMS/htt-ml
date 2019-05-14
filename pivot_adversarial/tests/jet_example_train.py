import numpy as np
import pickle
import sys
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams["font.size"] = 16.0

import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam

def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * K.categorical_crossentropy(y_pred, y_true)
    return loss_D


def make_loss_R(c):
    def loss_R(z_true, z_pred):
        return c * K.categorical_crossentropy(z_pred, z_true)
    return loss_R

def plot_losses(losses):
    ax1 = plt.subplot(311)
    values = np.array(losses["L_f"])
    plt.plot(range(len(values)), values, label=r"$L_f$", color="blue")
    plt.legend(loc="upper right")

    ax2 = plt.subplot(312, sharex=ax1)
    values = np.array(losses["L_r"])
    plt.plot(range(len(values)), values, label=r"$L_r$", color="green")
    plt.legend(loc="upper right")

    ax3 = plt.subplot(313, sharex=ax1)
    values = np.array(losses["L_f - L_r"])
    plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")
    plt.legend(loc="upper right")

    plt.savefig('losses-jets.png')

# Command line arguments
lam = float(sys.argv[1])
seed = int(sys.argv[2])
np.random.seed = seed

# Prepare data
fd = open("/home/sjoerger/workspace/pivot_adversarials/jet_example/jets-pile.pickle", "rb")
X_train, X_test, y_train, y_test, z_train, z_test, scaler = pickle.load(fd)
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train.astype(np.int))
y_test = np_utils.to_categorical(y_test.astype(np.int))
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
Dx = Dense(32, activation="relu")(inputs)
Dx = Dropout(0.2)(Dx)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dropout(0.2)(Dx)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dropout(0.2)(Dx)
Dx = Dense(2, activation="softmax")(Dx)
D = Model(inputs=inputs, outputs=Dx)

opt_D = Adam()
D.trainable = True
for layer in D.layers:
    layer.trainble= True
D.compile(loss=['categorical_crossentropy'], optimizer=opt_D, metrics=['accuracy'])

Rx_input = Input(shape=(2,))
Rx = Dense(64, activation="relu")(Rx_input)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(64, activation="relu")(Rx)
Rx = Dense(z.shape[1], activation="softmax")(Rx)
R = Model(inputs=Rx_input, outputs=Rx)

x = D(inputs)

an = Model(inputs=[inputs], outputs=[D(inputs),R(D(inputs))])
opt_an = SGD()
D.trainable = True
for layer in D.layers:
    layer.trainable = True
R.trainable = False
for layer in R.layers:
    layer.trainable = False
an.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1.,-lam], optimizer=opt_an, metrics=['accuracy'])
an.summary()

opt_R = SGD()
adv = Model(inputs=[inputs], outputs=R(D(inputs)))
D.trainable = False
for layer in D.layers:
    layer.trainable = False
R.trainable = True
for layer in R.layers:
    layer.trainable = True
adv.compile(loss=['categorical_crossentropy'], optimizer=opt_R, metrics=['accuracy'])
adv.summary()


# Pretraining of Classifier

batch_size = 128
D.trainable = True
for layer in D.layers:
    layer.trainable = True

D.fit(X_train, y_train, epochs=10)


# Pretraining of adversary
D.trainable = False
for layer in D.layers:
    layer.trainable = False
R.trainable = True
for layer in R.layers:
    layer.trainable = True

if lam > 0.0:
    adv.fit(X_train[(y_train[:,0] == 1)], z_train[(y_train[:,0] == 1)], epochs=10)


# Adversarial training
batch_size = 128
losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

print(an.evaluate(X_test, [y_test,z_test]))

for i in range(1001):
    print(i)

    if i % 10 == 0:
        loss = an.evaluate(X_test, [y_test, z_test], verbose=0)
        losses["L_f - L_r"].append(loss[0])
        losses["L_f"].append(loss[1])
        losses["L_r"].append(-loss[2])
        print('L_r: {}'.format(losses["L_r"][-1]))
        print('L_f: {}'.format(losses['L_f'][-1]))
        print('L_f - L_r: {}'.format(losses["L_f - L_r"][-1]))

    # Fit Adversary
    if lam > 0.0:
        D.trainable = False
        for layer in D.layers:
            layer.trainable = False
        R.trainable = True
        for layer in R.layers:
            layer.trainable = True

        adv.fit(X_train[(y_train[:,0] == 1)], z_train[(y_train[:,0] == 1)],
                batch_size=batch_size, verbose=1)

    # Fit combined network
    D.trainable = True
    for layer in D.layers:
        layer.trainable = True
    R.trainable = False
    for layer in R.layers:
        layer.trainable = False
    indices = np.random.permutation(len(X_train))[:batch_size]
    an.train_on_batch(X_train[indices], [y_train[indices], z_train[indices]])

print(an.evaluate(x=X_test, y=[y_test, z_test]))
plot_losses(losses)
D.save_weights("D-%.4f-%d-z=0.h5" % (lam, seed))

