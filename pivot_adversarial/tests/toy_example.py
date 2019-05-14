import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (6,6)
# plt.rcParams["font.size"] = 16.0

import numpy as np
import time
np.random.seed(333)

n_samples = 125000

X0 = np.random.multivariate_normal(mean=np.array([0., 0.]),  cov=np.array([[1., -0.5], [-0.5, 1.]]), size=n_samples // 2)
X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples //2)
z = np.random.normal(loc=0.0, scale=1.0, size=n_samples)


X1[:,1] += z[n_samples //2 :]

X = np.vstack([X0, X1])
y = np.zeros(n_samples)
y[n_samples // 2:] = 1

# plt.title("$X$")
# plt.scatter(X[y==0, 0], X[y==0, 1], c="r", marker="o", edgecolors="none")
# plt.scatter(X[y==1, 0], X[y==1, 1], c="b", marker="o", edgecolors="none")
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

X_train, X_valid, y_train, y_valid, z_train, z_valid = train_test_split(X, y, z, test_size=50000)

# print(X_train)
# print(y_train)
# print(z_train)

import keras.backend as K
from keras.layers import Input, Dense, Merge, Lambda, Concatenate, Activation
from keras.models import Model, Sequential
from keras.losses import binary_crossentropy
import tensorflow as tf

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

# def create_combined(discriminator, adversary, inputs, lam):
#     adversary.trainable=False
#     x = discriminator(inputs)
#     an_output = adversary(x)
#     an = Model(inputs=inputs, outputs=[x,an_output])
#     an.compile(loss=['binary_crossentropy', make_loss_R(lam=-lam)], optimizer='adam')
#     return an

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

from keras.optimizers import SGD, Adam
import keras.backend as K

lam = 50.0

def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * binary_crossentropy(y_pred, y_true)

    return loss_D


def make_loss_R(lam):
    def loss(y_true, y_pred):
        return lam * gnll_loss(y_true, y_pred)

    return loss

from keras.utils.generic_utils import get_custom_objects
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

def make_X(n_samples, z):
    np.random.seed(int(time.time()))
    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples // 2)
    X1[:, 1] += z
    X = np.vstack([X0, X1])
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1
    #
    # plt.title("$X$")
    # plt.scatter(X[y==0, 0], X[y==0, 1], c="r", marker="o", edgecolors="none")
    # plt.scatter(X[y==1, 0], X[y==1, 1], c="b", marker="o", edgecolors="none")
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    # plt.show()

    return X

z_minus_pred = discriminator.predict(make_X(200000, z=-1))
z_minus_pred = np.reshape(z_minus_pred, np.shape(z_minus_pred)[0])

z_zero_pred = discriminator.predict(make_X(200000, z=0))
z_zero_pred = np.reshape(z_zero_pred, np.shape(z_zero_pred)[0])

z_plus_pred = discriminator.predict(make_X(200000, z=1))
z_plus_pred = np.reshape(z_plus_pred, np.shape(z_plus_pred)[0])

plt.hist(z_minus_pred, bins=50, normed=1, histtype="step", label="$p(f(X)|Z=-\sigma)$")
plt.hist(z_zero_pred, bins=50, normed=1, histtype="step", label="$p(f(X)|Z=0)$")
plt.hist(z_plus_pred, bins=50, normed=1, histtype="step", label="$p(f(X)|Z=+\sigma)$")
plt.legend(loc="best")
#plt.ylim(0,4)
plt.xlabel("$f(X)$")
plt.ylabel("$p(f(X))$")
plt.grid()
plt.savefig("f-plain.png")
plt.savefig("f-plain.pdf")
plt.clf()

plt.clf()

fpr = dict()
tpr = dict()

fpr['plain'], tpr['plain'], _ = roc_curve(y_true=y_valid, y_score=discriminator.predict(X_valid))

from matplotlib.mlab import griddata

X_test = np.random.rand(30000, 2)
X_test[:, 0] *= 10.
X_test[:, 0] -= 5.
X_test[:, 1] *= 10.
X_test[:, 1] -= 5.

y_pred = discriminator.predict(X_test)

y_pred = np.reshape(y_pred, np.shape(y_pred)[0])

xi = np.linspace(-1., 2., 100)
yi = np.linspace(-1., 3, 100)
zi = griddata(x=X_test[:, 0], y=X_test[:, 1], z=y_pred, xi=xi, yi=yi, interp="linear")
CS = plt.contourf(xi, yi, zi, 20, cmap=plt.cm.viridis,
                  vmax=1.0, vmin=0.0)
plt.colorbar()
plt.scatter([0], [0], c="red", linewidths=0, label=r"$\mu_0$")
plt.scatter([1], [0], c="blue", linewidths=0, label=r"$\mu_1|Z=z$")
plt.scatter([1], [0+1], c="blue", linewidths=0)
plt.scatter([1], [0+2], c="blue", linewidths=0)
plt.text(1.2, 0-0.1, "$Z=-\sigma$", color="k")
plt.text(1.2, 1-0.1, "$Z=0$", color="k")
plt.text(1.2, 2-0.1, "$Z=+\sigma$", color="k")
plt.xlim(-1,2)
plt.ylim(-1,3)
plt.legend(loc="upper left", scatterpoints=1)
plt.savefig("surface-plain.png")
plt.savefig("surface-plain.pdf")
#plt.show()

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

    # R.trainable = True
    # for j in range(50):
    #     indices = np.random.permutation(len(X_train))
    #     d_train = discriminator.predict(X_train[indices])
    #     R.train_on_batch(d_train, z_train_padding[indices])
    #DfR.train_on_batch(X_train[indices], z_train_padding[indices])#, batch_size=batch_size, nb_epoch=1, verbose=1)


plot_losses(losses)

plt.clf()

plt.hist(discriminator.predict(make_X(200000, z=-1)), bins=50, normed=1, histtype="step", label="$p(f(X)|Z=-\sigma)$")
plt.hist(discriminator.predict(make_X(200000, z=0)), bins=50, normed=1, histtype="step", label="$p(f(X)|Z=0)$")
plt.hist(discriminator.predict(make_X(200000, z=1)), bins=50, normed=1, histtype="step", label="$p(f(X)|Z=+\sigma)$")
plt.legend(loc="best")
#plt.ylim(0,4)
plt.xlabel("$f(X)$")
plt.ylabel("$p(f(X))$")
plt.grid()
plt.savefig("f-adversary.png")
plt.savefig("f-adversary.pdf")
plt.clf()

fpr['adversary'], tpr['adversary'], _ = roc_curve(y_true=y_valid, y_score=discriminator.predict(X_valid))


from matplotlib.mlab import griddata

X_test = np.random.rand(30000, 2)
X_test[:, 0] *= 10.
X_test[:, 0] -= 5.
X_test[:, 1] *= 10.
X_test[:, 1] -= 5.

y_pred = discriminator.predict(X_test)

y_pred = np.reshape(y_pred, np.shape(y_pred)[0])

xi = np.linspace(-1., 2., 100)
yi = np.linspace(-1., 3, 100)
zi = griddata(x=X_test[:, 0], y=X_test[:, 1], z=y_pred, xi=xi, yi=yi, interp="linear")
CS = plt.contourf(xi, yi, zi, 20, cmap=plt.cm.viridis,
                  vmax=1.0, vmin=0.0)

plt.colorbar()
plt.scatter([0], [0], c="red", linewidths=0, label=r"$\mu_0$")
plt.scatter([1], [0], c="blue", linewidths=0, label=r"$\mu_1|Z=z$")
plt.scatter([1], [0+1], c="blue", linewidths=0)
plt.scatter([1], [0+2], c="blue", linewidths=0)
plt.text(1.2, 0-0.1, "$Z=-\sigma$", color="k")
plt.text(1.2, 1-0.1, "$Z=0$", color="k")
plt.text(1.2, 2-0.1, "$Z=\sigma$", color="k")
plt.xlim(-1,2)
plt.ylim(-1,3)
plt.legend(loc="upper left", scatterpoints=1)
plt.savefig("surface-adversary.png")
plt.savefig("surface-adversary.pdf")
plt.clf()

for key, item in fpr.items():
    fpr_lam = item
    tpr_lam = tpr[key]
    plt.plot(fpr_lam, tpr_lam, lw=2, label='{}'.format(key))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

output_path = 'roc_all_1'

plt.savefig(output_path + '.pdf')
plt.savefig(output_path + '.png')