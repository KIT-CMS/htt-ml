import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (6,6)
# plt.rcParams["font.size"] = 16.0

from keras.layers import Input, Dense, Merge, Lambda, Concatenate, Activation
from keras.models import Model, Sequential
from keras.losses import binary_crossentropy
import keras.backend as K

from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

import numpy as np
import time
from sklearn.model_selection import train_test_split

def create_discriminator(inputs, neurons):
    Dx = Dense(neurons, activation="relu")(inputs)
    Dx = Dense(neurons, activation='relu')(Dx)
    Dx = Dense(neurons, activation="relu")(Dx)
    Dx = Dense(1, activation="sigmoid")(Dx)
    D = Model(inputs=[inputs], outputs=[Dx])
    return D

def create_adversary(components, neurons, inputs):

    h1 = Dense(neurons, activation="relu", name="h1")(inputs)
    h2 = Dense(neurons, activation='relu', name='h2')(h1)
    h3 = Dense(neurons, activation="relu", name="h3")(h2)

    alphas = Dense(components, activation="softmax", name="alphas")(h3)
    mus = Dense(components, name="mus")(h3)
    sigmas = Dense(components, activation="nnelu", name="sigmas")(h3)
    pvec = Concatenate(name="pvec")([alphas, mus, sigmas])
    model = Model(inputs=inputs, outputs=pvec)
    return model

def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:, i * components:(i + 1) * components] for i in range(no_parameters)]


def gnll_loss(y_true, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector)  # Unpack parameter vectors

    y_true = y_true[:,0]

    n_components = mu.shape[1]

    pdf = alpha[:, 0] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, 0] *
                      K.exp(-(y_true - mu[:, 0]) ** 2 / (2. * sigma[:, 0] ** 2)))

    for c in range(1, n_components):
        pdf += alpha[:, c] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, c] *
                           K.exp(-(y_true - mu[:, c]) ** 2 / (2. * sigma[:, c] ** 2)))

    nll = -K.log(pdf)

    return K.mean(nll, axis=-1)

def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * binary_crossentropy(y_pred, y_true)

    return loss_D


def make_loss_R(lam):
    def loss(y_true, y_pred):
        return lam * gnll_loss(y_true, y_pred)

    return loss


np.random.seed(333)


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

    plt.savefig('losses.png')
    plt.savefig('losses.pdf')


n_samples = 125000

X0 = np.random.multivariate_normal(mean=np.array([0., 0.]),  cov=np.array([[1., -0.5], [-0.5, 1.]]), size=n_samples // 2)
X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples //2)
z = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

X1[:,1] += z[n_samples //2 :]

X = np.vstack([X0, X1])
y = np.zeros(n_samples)
y[n_samples // 2:] = 1

X_train, X_valid, y_train, y_valid, z_train, z_valid = train_test_split(X, y, z, test_size=50000)

lam = 50.0

get_custom_objects().update({'nnelu': Activation(nnelu)})

inputs_d = Input(shape=(X.shape[1],))
discriminator = create_discriminator(inputs=inputs_d, neurons=64)
discriminator.trainable = True
for layer in discriminator.layers:
    layer.trainable = True
discriminator.compile(loss=['binary_crossentropy'], optimizer='sgd', metrics=['accuracy'])
discriminator.summary()


no_parameters = 3
components = 5

inputs_r = Input(shape=(1,))

R = create_adversary(components=components, neurons=64, inputs=inputs_r)


combined = Model(inputs=[inputs_d], outputs=[discriminator(inputs_d), R(discriminator(inputs_d))])
discriminator.trainable = True
for layer in discriminator.layers:
    layer.trainable = True
R.trainable = False
for layer in R.layers:
    layer.trainable = False
combined.compile(loss=['binary_crossentropy', make_loss_R(-lam)], optimizer='sgd')
combined.summary()

adv = Model(inputs=[inputs_d], outputs=R(discriminator(inputs_d)))
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
R.trainable = True
for layer in R.layers:
    layer.trainable = True
adv.compile(loss=[make_loss_R(1.0)], optimizer='sgd')
adv.summary()


# Pretraining of D
discriminator.trainable = True
for layer in discriminator.layers:
    layer.trainable = True
discriminator.fit(X_train, y_train, epochs=50, shuffle=True)
min_Lf = discriminator.evaluate(X_valid, y_valid)
print(min_Lf)

discriminator.save('toy_gmm_normal.h5')

# Pad z_train and z_valid to fit the output nodes of the model?

z_train_reshape = np.reshape(z_train, (-1,1))
z_train_padding = np.zeros((z_train_reshape.shape[0], no_parameters*components))
z_train_padding[:z_train_reshape.shape[0],:z_train_reshape.shape[1]] = z_train_reshape

z_valid_reshape = np.reshape(z_valid, (-1,1))
z_valid_padding = np.zeros((z_valid_reshape.shape[0], no_parameters*components))
z_valid_padding[:z_valid_reshape.shape[0],:z_valid_reshape.shape[1]] = z_valid_reshape

# Pretraining of R
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
R.trainable = True
for layer in R.layers:
    layer.trainable = True
min_Lr = adv.evaluate(X_train, z_train_padding)
adv.fit(X_train, z_train_padding, epochs=20, shuffle=True)

print(min_Lr)

losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

batch_size = 128

for i in range(201):
    l = combined.evaluate(X_valid, [y_valid, z_valid_padding], verbose=0)
    losses["L_f - L_r"].append(l[0])
    losses["L_f"].append(l[1])
    losses["L_r"].append(-l[2]/lam)
    print('L_r: {}'.format(losses["L_r"][-1]))
    print('L_f: {}'.format(losses['L_f'][-1]))
    print('L_f - L_r: {}'.format(losses["L_f - L_r"][-1]))

    if i % 10 == 0:
        print('Currently on batch # {}'.format(i))



    # Fit D
    discriminator.trainable = True
    for layer in discriminator.layers:
        layer.trainable = True
    R.trainable = False
    for layer in R.layers:
        layer.trainable = False
    indices = np.random.permutation(len(X_train))[:batch_size]
    combined.train_on_batch(X_train[indices], [y_train[indices], z_train_padding[indices]])

    # Fit R
    discriminator.trainable = False
    for layer in discriminator.layers:
        layer.trainable = False
    R.trainable = True
    for layer in R.layers:
        layer.trainable = True
    adv.fit(X_train, z_train_padding, batch_size=batch_size, epochs=1, verbose=1)

plot_losses(losses)
discriminator.save('toy_gnn_pivot.h5')