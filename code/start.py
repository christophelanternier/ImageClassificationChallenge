import numpy as np
from numpy import genfromtxt
from classification import *
from kernel import *
from datetime import time, datetime, timedelta

# store data
Xtr = genfromtxt('../data/Xtr.csv', delimiter=',')
Ytr = genfromtxt('../data/Ytr.csv', delimiter=',')
#Xte = genfromtxt('../data/Xte.csv', delimiter=',')

Xtr = np.delete(Xtr, 3072, axis=1)
#Xte = np.delete(Xte, 3072, axis=1)
Ytr = Ytr[1:,1]
N = len(Ytr)

N_train = 3000
train_images = Xtr[:N_train,:]
train_labels = Ytr[:N_train]

test_images = Xtr[N_train:,:]
test_labels = Ytr[N_train:]

def predict_with_scattering_kernel_and_SVM():
    orders_scales = [(2, 3)]
    lambdas = [0.01]

    for (order, scale) in orders_scales:
        t0 = datetime.now()
        print 'computing features with order ', order, ' and scale ', scale
        train_scattering_features = scattering_kernel(train_images, order, scale).T
        test_scattering_features = scattering_kernel(test_images, order, scale).T
        t1 = datetime.now()
        print 'done. duration : ', t1 - t0

        for _lambda in lambdas:
            t1 = datetime.now()
            alphas, bias = one_versus_all_SVM(train_scattering_features, train_labels, _lambda=_lambda)
            t2 = datetime.now()
            print 'model fitted. duration: ', t2 - t1
            prediction = predict_SVM(alphas, bias, train_scattering_features, test_scattering_features)
            t3 = datetime.now()
            print 'prediction done. duration: ', t3 - t2

            if test_labels is not None:
                well_classified = 0
                for i in range(len(prediction)):
                    if prediction[i] == test_labels[i]:
                        well_classified += 1
                print 'lambda = ', _lambda, ', order = ', order, ' scale = ', scale, ' rate = ', float(well_classified) / len(prediction)
            else:
                DF = pd.DataFrame(data=pd.Series(prediction), columns=['Prediction'])
                DF.index += 1
                DF.to_csv('../data/'+'scattering_transform_order_2_scale_3.csv', index=True, index_label='Id', sep=',')

def predict_with_scattering_kernel_and_class_PCA():
    orders_scales = [(2, 3)]

    for (order, scale) in orders_scales:
        t0 = datetime.now()
        print 'computing features with order ', order, ' and scale ', scale
        train_scattering_features = scattering_kernel(train_images, order, scale)
        print 'train features shape ', train_scattering_features.shape
        test_scattering_features = scattering_kernel(test_images, order, scale)
        print 'test features shape ', test_scattering_features.shape
        t1 = datetime.now()
        print 'done. duration : ', t1 - t0

        t1 = datetime.now()
        means, projection_basis = compute_class_PCA_linear_space(train_scattering_features, train_labels)
        t2 = datetime.now()
        print 'projection basis computed. durartion : ', t2 - t1

        PCA_space_dimensions = range(5, 30, 1)
        for dimension in PCA_space_dimensions:
            t3 = datetime.now()
            predicted_labels = predict_with_class_PCA_projection(test_scattering_features, means, projection_basis, dimension)
            t4 = datetime.now()
            print 'labels predicted. duration: ', t4 - t3

            if test_labels is not None:
                well_classified = 0
                for i in range(predicted_labels.size):
                    if predicted_labels[i] == test_labels[i]:
                        well_classified += 1
                print 'dimension = ', dimension, ', order = ', order, ' scale = ', scale, ' rate = ', float(well_classified) / predicted_labels.size
            else:
                DF = pd.DataFrame(data=pd.Series(predicted_labels), columns=['Prediction'])
                DF.index += 1
                DF.to_csv('../data/'+'scattering_transform_order_'+str(order)+'_scale_'+str(scale)+'_dim_'+str(dimension)+'.csv', index=True, index_label='Id', sep=',')

def predict_with_first_scattering_kernel():
    _lambda = 0.01
    print 'computing features ...'
    train_features = first_scattering_kernel(train_images).T
    test_features = first_scattering_kernel(test_images).T
    print '... done'

    t1 = datetime.now()
    alphas, bias = one_versus_all_SVM(train_features, train_labels, _lambda=_lambda)
    t2 = datetime.now()
    print 'model fitted. duration: ', t2 - t1
    prediction = predict_SVM(alphas, bias, train_features, test_features)
    t3 = datetime.now()
    print 'prediction done. duration: ', t3 - t2

    if test_labels is not None:
        well_classified = 0
        for i in range(len(prediction)):
            if prediction[i] == test_labels[i]:
                well_classified += 1
        print 'lambda = ', _lambda, ', good classification rate = ', float(well_classified) / len(prediction)
    else:
        df = pd.DataFrame(data=pd.Series(prediction), columns=['prediction'])
        df.index += 1
        df.to_csv('../data/'+'scattering_transform_2.csv', index=True, index_label='id', sep=',')

predict_with_scattering_kernel_and_class_PCA()
