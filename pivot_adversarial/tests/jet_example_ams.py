import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,6)

import numpy as np
np.random.seed = 777

import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
import pickle

fd = open("/home/sjoerger/workspace/pivot_adversarials/jet_example/jets-pile.pickle", "rb")
X_train, X_test, y_train, y_test, z_train, z_test, scaler = pickle.load(fd)
X = X_test
y = y_test
z = z_test

lam = 1.0

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

def plot_AMS1(pred,Y,Z, label):
    x00 = np.sort(pred[((Y==0)*(Z==0))].flatten())
    x01 = np.sort(pred[((Y==0)*(Z==1))].flatten())
    x10 = np.sort(pred[((Y==1)*(Z==0))].flatten())
    #print(pred)
    n_points = 100
    AMS1 = np.zeros(n_points)
    all_ns = np.zeros(n_points)
    all_nb = np.zeros(n_points)
    all_nb1 = np.zeros(n_points)
    all_sigb = np.zeros(n_points)

    cuts = np.zeros(n_points)
    sig_eff = np.zeros(n_points)
    ns_tot = x10.shape[0]
    
    for i, c_i in enumerate(np.linspace(0.0, 1.0, n_points)):
        cuts[i] = c_i
        #print(c_i, np.count_nonzero(x10 > c_i), (100. / x10.size))
        ns = (100. / x10.size) * np.count_nonzero(x10 > c_i)
        nb = (1000. / x00.size) * np.count_nonzero(x00 > c_i)
        nb1 = (1000. / x01.size) * np.count_nonzero(x01 > c_i)
        sig_b = 1.0 * np.abs(nb - nb1) 
        
        print(ns, nb, nb1, sig_b)

        b0 = 0.5 * (nb - sig_b ** 2 + ((nb - sig_b ** 2) ** 2 + 4 * (ns + nb) * (sig_b ** 2)) ** 0.5)
        # if b0 < 1e-7 or sig_b < 1e-7:
        #     AMS1[i] = 0
        # else:
        AMS1[i] = (2.0 * ((ns + nb) * np.log((ns + nb) / b0) - ns - nb + b0) + ((nb - b0) / sig_b) ** 2) ** 0.5
        
        all_ns[i] = ns
        all_nb[i] = nb
        all_nb1[i] = nb
        all_sigb[i] = sig_b
        
        sig_eff[i] = (1.0*ns) / ns_tot
        
    return cuts, AMS1

import glob

indices = np.random.permutation(len(X))
indices = indices[:5000000]

for lam, c in zip([0.0], ["r"]):
    ams = []

    for f in glob.glob("D-%.4f-*.h5" % lam):
        print(f)
        D.load_weights(f)
        cuts, a = plot_AMS1(D.predict(X[indices]), y[indices], z[indices, 1], f)
        ams.append(a)

    #print(cuts, ams[0])
    mu = np.mean(ams, axis=0) 
    std = np.std(ams, axis=0)
    plt.plot(cuts, ams[0], label=r"$\lambda=%d|Z=0$" % lam, c=c)
    #plt.fill_between(cuts, mu+std, mu-std, color=c, alpha=0.1)
    
for lam, c in zip([0.0, 1.0, 10, 500.], ["g", "b", "c", "orange", "y", "grey", "k"]):
    ams = []

    for f in glob.glob("y=0-D-%.4f-*.h5" % lam):
        print(f)
        D.load_weights(f)
        cuts, a = plot_AMS1(D.predict(X[indices]), y[indices], z[indices, 1], f)
        ams.append(a)
        
    mu = np.mean(ams, axis=0) 
    std = np.std(ams, axis=0)
    plt.plot(cuts, ams[0], label=r"$\lambda=%d$" % lam, c=c)
    #plt.fill_between(cuts, mu+std, mu-std, color=c, alpha=0.1)
    
plt.legend(loc="best")
plt.ylabel("AMS")
plt.xlabel("threshold on $f(X)$")
plt.grid()
plt.savefig("/home/sjoerger/workspace/pivot_adversarials/jet_example/ams.png")
plt.show()
