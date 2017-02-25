import numpy as np
from numpy import genfromtxt
from classification import *
from kernel import *
from datetime import time, datetime, timedelta

# store data
Xtr = genfromtxt('../data/Xtr.csv', delimiter=',')
Ytr = genfromtxt('../data/Ytr.csv', delimiter=',')
Xte = genfromtxt('../data/Xte.csv', delimiter=',')

Xtr = np.delete(Xtr, 3072, axis=1)
Xte = np.delete(Xte, 3072, axis=1)
Ytr = Ytr[1:,1]
N = len(Ytr)

N_train = 3000
train_images = Xtr[0:N_train,:]
train_labels = Ytr[0:N_train,:]

test_images = Xtr[N_train:,:]
test_labels = Ytr[N_train:,:]

maximum_scales = [2, 3, 4]
lambdas = [0.1, 0.01, 0.001]

for maximum_scale in maximum_scales:
    train_scattering_features = scattering_kernel(train_images, maximum_scale=maximum_scale).T
    test_scattering_features = scattering_kernel(test_images, maximum_scale=maximum_scale).T

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
                print 'lambda = ', _lambda, ', good classification rate = ', float(well_classified) / (N - N_train)
        else:
            DF = pd.DataFrame(data=pd.Series(prediction), columns=['Prediction'])
            DF.index += 1
            DF.to_csv('../data/'+'scattering_transform_lambda_' + str(_lambda) + '.csv', index=True, index_label='Id', sep=',')
