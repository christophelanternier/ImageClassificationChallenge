import numpy as np
from numpy import genfromtxt
import pandas as pd
import classification as clf
import kernel as ker
from datetime import datetime


def predict_with_SVM(train_features, train_labels, test_features, test_labels=None, lambdas=[0.01],
                     filename='../data/Y_te_SVM.csv',Kernel=ker.linear_kernel):
    for _lambda in lambdas:
        t1 = datetime.now()
        alphas, bias = clf.one_versus_all_SVM(train_features, train_labels, _lambda=_lambda,Kernel=Kernel)
        t2 = datetime.now()
        print 'model fitted. duration: ', t2 - t1
        predicted_labels = clf.predict_SVM(alphas, bias, train_features, test_features,Kernel=Kernel)
        t3 = datetime.now()
        print 'prediction done. duration: ', t3 - t2

        if test_labels is not None:
            well_classified = (predicted_labels == test_labels).sum()
            print 'lambda = ', _lambda, ' rate = ', float(well_classified) / len(predicted_labels)
        else:
            DF = pd.DataFrame(data=pd.Series(predicted_labels), columns=['Prediction'])
            DF.index += 1
            DF.to_csv(filename, index=True, index_label='Id', sep=',')


def predict_with_class_PCA(train_features, train_labels, test_features, test_labels=None,
                           PCA_dimensions=range(10, 35, 2), filename='../data/Y_te_PCA.csv'):
    t1 = datetime.now()
    means, projection_basis = clf.compute_class_PCA_linear_space(train_features, train_labels)
    t2 = datetime.now()
    print 'projection basis computed. duration : ', t2 - t1

    # after cross val, dimension 22/34 seem to be the best
    for dimension in PCA_dimensions:
        t3 = datetime.now()
        predicted_labels = clf.predict_with_class_PCA_projection(test_features, means, projection_basis, dimension)
        t4 = datetime.now()
        print 'labels predicted. duration: ', t4 - t3

        if test_labels is not None:
            well_classified = (predicted_labels == test_labels).sum()
            print 'dimension = ', dimension, ' rate = ', float(well_classified) / predicted_labels.size
        else:
            DF = pd.DataFrame(data=pd.Series(predicted_labels), columns=['Prediction'])
            DF.index += 1
            DF.to_csv(filename, index=True, index_label='Id', sep=',')


Xtr = genfromtxt('../data/Xtr.csv', delimiter=',')
Xte = genfromtxt('../data/Xte.csv', delimiter=',')
Ytr = genfromtxt('../data/Ytr.csv', delimiter=',')

Xtr = np.delete(Xtr, 3072, axis=1)
Xte = np.delete(Xte, 3072, axis=1)
Ytr = Ytr[1:, 1]

N_train = 3000

train_images = Xtr  # [:N_train,:]
train_labels = Ytr  # [:N_train]
test_images = Xte  # Xtr[N_train:,:]
test_labels = None  # Ytr[N_train:]

train_features = ker.first_scattering_kernel(train_images)
test_features = ker.first_scattering_kernel(test_images)

predict_with_SVM(train_features, train_labels, test_features, test_labels,
                 lambdas=[0.001],
                 Kernel=ker.get_kernel(type='gaussian',sigma=.3),
                 filename='Yte.csv')
# predict_with_class_PCA(train_features, train_labels, test_features, test_labels)
