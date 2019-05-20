import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from keras.layers import Input, Dense, Merge, Lambda, Concatenate, Activation
from keras.models import Model, Sequential

import numpy as np
import time
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

    plt.savefig('losses_2.png')
    plt.savefig('losses_2.pdf')

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
    output = Dense(bins, activation='softmax', name='softmax')(h3)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def make_X(n_samples, z):
    np.random.seed(int(time.time()))
    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples // 2)
    X1[:, 1] += z
    X = np.vstack([X0, X1])
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1

    return X

n_samples = 125000
lam = 50.0

X0 = np.random.multivariate_normal(mean=np.array([0., 0.]),  cov=np.array([[1., -0.5], [-0.5, 1.]]), size=n_samples // 2)
X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples //2)
z = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

X1[:,1] += z[n_samples //2 :]

X = np.vstack([X0, X1])
y = np.zeros(n_samples)
y[n_samples // 2:] = 1

bins = 9

Z = np.zeros((n_samples, bins))
cuts = np.linspace(-2.0, 2.0, bins)
for i, c_i in enumerate(cuts):
    for j,value in enumerate(z):
        if i == 0:
            if value <= c_i:
                Z[j,i] = 1
        else:
            if value <= c_i and value > cuts[i-1]:
                Z[j,i] = 1


X_train, X_valid, y_train, y_valid, z_train, z_valid = train_test_split(X, y, Z, test_size=50000)

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
loss_weights = [1.0, -lam]
combined.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], loss_weights=loss_weights, optimizer='sgd')
combined.summary()

adv = Model(inputs=[inputs_d], outputs=R(discriminator(inputs_d)))
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
R.trainable = True
for layer in R.layers:
    layer.trainable = True
adv.compile(loss=['categorical_crossentropy'], optimizer='sgd', metrics=['accuracy'])
adv.summary()

# Pretraining of D
discriminator.trainable = True
for layer in discriminator.layers:
    layer.trainable = True
discriminator.fit(X_train, y_train, epochs=50, shuffle=True)
min_Lf = discriminator.evaluate(X_valid, y_valid)
print(min_Lf)

discriminator.save('toy_discriminator_normal.h5')

# Pretraining of R
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
R.trainable = True
for layer in R.layers:
    layer.trainable = True
min_Lr = adv.evaluate(X_train, z_train)
adv.fit(X_train, z_train, epochs=20, shuffle=True)

print(min_Lr)

losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

batch_size = 128

for i in range(201):
    l = combined.evaluate(X_valid, [y_valid, z_valid], verbose=0)
    losses["L_f - L_r"].append(l[0])
    losses["L_f"].append(l[1])
    losses["L_r"].append(l[2])
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
    combined.train_on_batch(X_train[indices], [y_train[indices], z_train[indices]])

    # Fit R
    discriminator.trainable = False
    for layer in discriminator.layers:
        layer.trainable = False
    R.trainable = True
    for layer in R.layers:
        layer.trainable = True
    adv.fit(X_train, z_train, batch_size=batch_size, epochs=1, verbose=1)


plot_losses(losses)

discriminator.save('toy_discriminator_pivot.h5')