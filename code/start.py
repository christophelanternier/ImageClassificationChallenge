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
N_train = 100
N = 150
N

train_images = Xtr[:N_train]
train_labels = Ytr[:N_train]

test_images = Xtr[N_train:N]
test_labels = Ytr[N_train:N]

def train_one_versus_all_logistic_classifier(images, labels):
    N = len(labels)
    alphas = np.zeros((n_classes, N))
    for c in classes:
        one_versus_all_labels = np.zeros(N)
        for i in range(N_train):
            if train_labels[i] == c:
                one_versus_all_labels[i] = 1
            else:
                one_versus_all_labels[i] = -1

        alphas[c, :] = logistic_regression_classifier(train_images, one_versus_all_labels)
        print "classifier for class ", c, " done"
    return alphas

def test_one_versus_all_logistic_classifier(alphas, train_images, test_images, test_labels):
    N = len(test_labels)
    error_rate = 0
    for i in range(N):
        test_image = test_images[i,:]
        max_probability = 0
        for c in classes:
            p_c = sigma(predict(alphas[c,:], train_images, test_image))
            if (p_c > max_probability):
                predicted_class = c
                max_probability = p_c

        if (c != test_labels[i]):
            error_rate += 1.0

    return error_rate / N

alphas = train_one_versus_all_logistic_classifier(train_images, train_labels)
error_rate = test_one_versus_all_logistic_classifier(alphas, train_images, test_images, test_labels)
print "error rate :", error_rate





