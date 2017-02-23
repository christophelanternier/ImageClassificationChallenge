import numpy as np
from numpy import genfromtxt
from classification import *
from kernel import *

# store data
Xtr = genfromtxt('../data/Xtr.csv', delimiter=',')
Ytr = genfromtxt('../data/Ytr.csv', delimiter=',')
Xte = genfromtxt('../data/Xte.csv', delimiter=',')

Xtr = np.delete(Xtr, 3072, axis=1)
Xte = np.delete(Xte, 3072, axis=1)
Ytr = Ytr[1:,1]
N = len(Ytr)

# chaque ligne est un exemple
#train_images = fourier_1D_kernel(Xtr)
train_images = wavelet_transform(Xtr)
train_labels = Ytr

test_images = wavelet_transform(Xte)

alphas = train_one_versus_all_logistic_classifier(train_images, train_labels)
n_labels = 10
test_one_versus_all_logistic_classifier(alphas, train_images, test_images, n_labels, test_labels=None, filename='../data/Yte_fourier_1D_log_reg.csv')
