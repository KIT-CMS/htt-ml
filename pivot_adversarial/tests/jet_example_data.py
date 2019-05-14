import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,6)

import numpy as np
np.random.seed = 777

import h5py

# Get data from http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/highlevel/

f = h5py.File("/home/sjoerger/workspace/pivot_adversarials/jet_example/test_no_pile_5000000.h5", "r")
X_no_pile = f["features"].value
y_no_pile = f["targets"].value.ravel()

f = h5py.File("/home/sjoerger/workspace/pivot_adversarials/jet_example/test_pile_5000000.h5", "r")
X_pile = f["features"].value
y_pile = f["targets"].value.ravel()


from sklearn.model_selection import train_test_split

X = np.vstack((X_no_pile, X_pile))
y = np.concatenate((y_no_pile, y_pile)).ravel()
z = np.zeros(len(X))
z[len(X_no_pile):] = 1

strates = np.zeros(len(X))
strates[(y==0)&(z==0)]=0
strates[(y==0)&(z==1)]=1
strates[(y==1)&(z==0)]=2
strates[(y==1)&(z==1)]=3

from keras.utils import np_utils
z = np_utils.to_categorical(z.astype(np.int))

from sklearn.preprocessing import StandardScaler
tf = StandardScaler()
X = tf.fit_transform(X)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=25000, random_state=1, stratify=strates)


X_train = X_train[:150000]
y_train = y_train[:150000]
z_train = z_train[:150000]

import pickle
with open('/home/sjoerger/workspace/pivot_adversarials/jet_example/jets-pile.pickle', 'wb') as file:
    pickle.dump([X_train, X_test, y_train, y_test, z_train, z_test, tf], file)

print('Sucessfully pickled the training data.')
