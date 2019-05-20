import matplotlib as mpl

mpl.use('Agg')
#mpl.rcParams['lines.linewidth'] = 2

import matplotlib.pyplot as plt

import numpy as np
import time
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import load_model
np.random.seed(333)

def make_X(n_samples, z):
    np.random.seed(int(time.time()))
    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples // 2)
    X1[:, 1] += z
    X = np.vstack([X0, X1])
    y0 = np.zeros(n_samples //2)
    y1 = np.ones(n_samples //2)

    return X0, X1

def prepare_data(x0, x1, discriminator):
    pred = discriminator.predict(x0)
    pred_signal = [i for i in pred if i <= 0.5]
    pred_background_0 = [i for i in pred if i > 0.5]

    pred = discriminator.predict(x1)
    pred_signal_1 = [i for i in pred if i > 0.5]
    pred_background = [i for i in pred if i <= 0.5]

    return pred_signal, pred_background

def make_shifted_examples(n_samples, z):
    np.random.seed(int(time.time()))
    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),
                                       size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2), size=n_samples // 2)
    X1[:, 1] += z
    X = np.vstack([X0, X1])
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1

    return X, y, z

def plot_significance(discriminator, title):
    nbins = 50

    fig, (ax, ax_2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))

    X0_nom, X1_nom = make_X(n_samples=400000, z=0)
    pred_0_nom = discriminator.predict(X0_nom)
    pred_1_nom = discriminator.predict(X1_nom)

    #print(pred_1_nom.ravel(), pred_0_nom.ravel())

    values_nominal, _, _ = ax.hist([pred_0_nom.ravel(), pred_1_nom.ravel()], bins=nbins, stacked=True, normed=0, histtype="step",
                                label=[r"$p(f(S))$", r"$p(f(B)|Z=0)$"], linewidth=3.0)

    X0_down, X1_down = make_X(n_samples=400000, z=-1)
    pred_1_down = discriminator.predict(X1_down)
    combined_down = np.vstack([pred_0_nom, pred_1_down])

    values_down, _, _ = ax.hist(combined_down, bins=nbins, normed=0, histtype="step",
                                label=r"$p(f(B)|Z=-\sigma)$", linewidth=3.0)


    X0_up, X1_up = make_X(n_samples=400000, z=+1)
    pred_1_up = discriminator.predict(X1_up)
    combined_up = np.vstack([pred_0_nom, pred_1_up])
    values_up, _, _ = ax.hist(combined_up, bins=nbins, normed=0,
                                        histtype="step",
                                        label= r"$p(f(B)|Z=+\sigma)$",
                                        linewidth=3.0)

    ax.legend(loc="best")
    ax.set_xlim([0., 1.0])
    ax.set_ylim([0.,60000.])
    plt.xlabel("$f(X)$")
    ax.set_ylabel("$Number of events$")
    plt.grid()

    background_values = values_nominal[1] - values_nominal[0]

    sigma_b = []
    significance = []
    x_values = []
    for i, bin in enumerate(values_down):
        sigma_b.append(np.abs((values_up[i] - bin) / 2.))

    for i, bin in enumerate(values_nominal[0]):
        signal_events = bin
        background_events = background_values[i]
        significance.append(signal_events / np.sqrt(signal_events + background_events + sigma_b[i]))
        x_values.append(float(i) / float(nbins) + 1. / float(nbins) / 2.)

    ax_2.plot(x_values, significance, 'ro')
    ax_2.set_xlim([0., 1.0])
    ax_2.set_ylabel(r'$S/\sqrt{S+B + \sigma_B}$')
    ax_2.set_ylim([0.,100.])
    fig.tight_layout()
    fig.savefig(title + ".png", bbox_inches="tight")
    fig.savefig(title + ".pdf", bbox_inches="tight")
    fig.clf()
    pass


discriminator = load_model('toy_discriminator_normal.h5')

plot_significance(discriminator, title='normal_softmax')

plt.clf()

tpr_plain = dict()
fpr_plain = dict()
auc_plain = dict()

for shift in [1, 0, -1]:
    X_roc, y_roc, z_roc = make_shifted_examples(n_samples=200000, z=shift)
    fpr_plain[shift], tpr_plain[shift], _ = roc_curve(y_true=y_roc, y_score=discriminator.predict(X_roc))
    auc_plain[shift] = roc_auc_score(y_true=y_roc, y_score=discriminator.predict(X_roc))


plt.clf()

discriminator = load_model('toy_discriminator_pivot.h5')

plot_significance(discriminator, title='pivot_softmax')

plt.clf()

tpr_adversary = dict()
fpr_adversary = dict()
auc_adversary = dict()

plt.figure(figsize=(10,8))

for shift in [1, 0, -1]:
    X_roc, y_roc, z_roc = make_shifted_examples(n_samples=200000, z=shift)
    fpr_adversary[shift], tpr_adversary[shift], _ = roc_curve(y_true=y_roc, y_score=discriminator.predict(X_roc))
    auc_adversary[shift] = roc_auc_score(y_true=y_roc, y_score=discriminator.predict(X_roc))

for key, item in fpr_plain.items():
    fpr_lam = item
    tpr_lam = tpr_plain[key]
    AUC = auc_plain[key]
    plt.plot(fpr_lam, tpr_lam, lw=3, label='Plain: Shift={}, AUC={}'.format(key, AUC))
for key, item in fpr_adversary.items():
    fpr_lam = item
    tpr_lam = tpr_adversary[key]
    AUC = auc_adversary[key]
    plt.plot(fpr_lam, tpr_lam, lw=3, label='Adversary: Shift={}, AUC={}'.format(key, AUC))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

output_path = 'roc-all'

plt.savefig(output_path + '.pdf')
plt.savefig(output_path + '.png')