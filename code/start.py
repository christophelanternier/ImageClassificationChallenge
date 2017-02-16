import numpy as np
from numpy import genfromtxt
from classification import *

# store data
Xtr = genfromtxt('../data/Xtr.csv', delimiter=',')
Ytr = genfromtxt('../data/Ytr.csv', delimiter=',')
Xtr = np.delete(Xtr, 3072, axis=1)
Ytr = Ytr[1:,1]
N = len(Ytr)
classes = [int(c) for c in set(Ytr)]
n_classes = len(classes)

# divide data in train and test for validation
N_train = 2 * N / 3
N_test = N - N_train

train_images = Xtr[:N_train]
train_labels = Ytr[:N_train]

test_images = Xtr[N_train:N]
test_labels = Ytr[N_train:N]


alphas = train_one_versus_all_logistic_classifier(train_images, train_labels)
n_labels = 10
test_one_versus_all_logistic_classifier(alphas, train_images, test_images, n_labels, test_labels)





