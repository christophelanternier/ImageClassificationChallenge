import cvxopt
import numpy as np
import pandas as pd
from utils import *

#####################
# PCA
####################

def compute_class_PCA_linear_space(features, labels):
    n_labels = np.unique(labels).size
    means = []
    projection_basis = []

    for label in range(n_labels):
        print 'computing PCA basis for label ', label, '...'
        label_indices = np.where(labels == label)[0]
        label_features = features[label_indices, :]
        print 'label features shape : ', label_features.shape
        mean, centered_features = recenter(label_features)
        covariance_matrix = np.cov(centered_features.T)
        print 'covariance_matrix shape : ', covariance_matrix.shape
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        decreasing_eigenvalues_index = eigenvalues.argsort()[::-1]
        print 'first 5 eigenvalues : ', eigenvalues[decreasing_eigenvalues_index][0:5]
        eigenvectors = eigenvectors[:,decreasing_eigenvalues_index]
        print 'eigenvectors shape : ', eigenvectors.shape

        means.append(mean)
        projection_basis.append(eigenvectors)
        print '...done'

    print 'n means : ', len(means), ' n projection basis ', len(projection_basis)
    return means, projection_basis

def predict_with_class_PCA_projection(features, means, projection_basis, dim):
    n_features = features.shape[0]

    for (label, (mean, basis)) in enumerate(zip(means, projection_basis)):
        centered_features = np.copy(features)
        for i in range(n_features):
            centered_features[i,:] = centered_features[i,:] - mean

        basis = basis[:, dim:].T
        print 'basis shape : ', basis.shape
        centered_features = centered_features.T
        print 'centered_features shape : ', centered_features.shape
        distances_to_PCA_space = np.linalg.norm(basis.dot(centered_features), axis=0)
        print 'distances_to_PCA_space shape : ', distances_to_PCA_space.shape

        if label == 0:
            predicted_labels = np.zeros(n_features)
            lowest_distance_to_PCA_space = distances_to_PCA_space
        else:
            for i in range(n_features):
                if distances_to_PCA_space[i] < lowest_distance_to_PCA_space[i]:
                    predicted_labels[i] = label
                    lowest_distance_to_PCA_space[i] = distances_to_PCA_space[i]
    return predicted_labels

#####################
#NEW SVM
####################

def one_versus_all_SVM(features, labels, _lambda):
    N = len(labels)
    n_labels = len(set(labels))
    alphas = np.zeros((n_labels, N))
    bias = np.zeros(n_labels)

    #Linear Kernel:
    K = features.T.dot(features)

    for label in range(n_labels):
        one_versus_all_labels = np.zeros(N)
        for i in range(N):
            if labels[i] == label:
                one_versus_all_labels[i] = 1
            else:
                one_versus_all_labels[i] = -1
        alphas[label, :], bias[label] = train_SVM(K, one_versus_all_labels, _lambda)
        print "classifier for label ", label, " done"

    return alphas, bias

def predict_SVM(alphas, bias, features, X):
    y_pred = np.zeros(alphas.shape[0])
    values_pred = np.zeros((alphas.shape[0],X.shape[1]))
    for k in range(alphas.shape[0]):
        values_pred[k,:] = alphas[k,:].dot(features.T.dot(X))+bias[k]
    return np.argmax(values_pred, axis=0)

def train_SVM(K, y, _lambda):

    n = y.shape[0]
    gamma = 1 / (2 * _lambda * n)

    P = cvxopt.matrix(K)

    h = cvxopt.matrix(0., (2 * n, 1))
    h[:n] = gamma

    A = cvxopt.matrix(1., (1, n))
    b = cvxopt.matrix(0.)

    y = y.astype(np.double)
    diag_y = cvxopt.spdiag(y.tolist())
    q = cvxopt.matrix(-y)
    G = cvxopt.sparse([diag_y, -diag_y])

    res = cvxopt.solvers.qp(P, q, G, h, A, b)

    return np.array(res["x"]).T, res["y"][0]

###################
#END OF NEW SVM
#################


def sigma(u):
    return 1.0 / (1 + np.exp(-u))

def logistic_loss(f_x, y):
    return np.log(1 + np.exp(-y * f_x))

def logistic_loss_prime(f_x, y):
    return - 1.0 / (1 + np.exp(y * f_x))

def logistic_regression_loss_function(alpha, K, labels, _lambda):
    N = len(alpha)
    f_x = K.dot(alpha)
    result = 0

    for i in range(0,N):
        f_x_i =f_x[i]
        y_i = labels[i]
        result += logistic_loss(f_x_i, y_i)

    result = result / N
    result += _lambda * alpha.dot(f_x)

    return result

def grad_logistic_regression_loss_function(alpha, K, labels, _lambda):
    N = len(alpha)
    P_diag = np.zeros(N)
    f_x = K.dot(alpha)

    for i in range(0,N):
        P_diag[i] = logistic_loss_prime(f_x[i], labels[i])

    Py = P_diag * labels

    return 1.0 / N * K.dot(Py) + _lambda * f_x

def compute_K(f_values):
    N = len(f_values[:,0])
    K = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            K[i, j] = f_values[i,:].dot(f_values[j,:])
    return K

def logistic_regression_classifier(f_values, labels, _lambda=0):
    # kernel logistic regression solved by gradient descent
    N = len(f_values[:,0])
    K = compute_K(f_values)
    alpha_0 = np.zeros(N)

    default_step = 10000
    stop_criterion = 10**(-1)
    gradient_norm = stop_criterion + 1
    n_iteration_max = 300
    iter = 0

    while (iter < n_iteration_max) and (gradient_norm > stop_criterion):
        step = default_step
        L = logistic_regression_loss_function(alpha_0, K, labels, _lambda)
        grad_L = grad_logistic_regression_loss_function(alpha_0, K, labels, _lambda)
        gradient_norm = np.linalg.norm(grad_L) / len(grad_L)
        L_update = logistic_regression_loss_function(alpha_0 - step * grad_L, K, labels, _lambda)
        while (L_update > L):
            step *= 0.5
            L_update = logistic_regression_loss_function(alpha_0 - step * grad_L, K, labels, _lambda)

        alpha_0 = alpha_0 - step * grad_L
        iter +=1

    if (iter == n_iteration_max):
        print "max iteration number reached"
    return alpha_0

def predict(alpha, f_values, x):
    f_x  = 0
    for i in range(0, len(alpha)):
        f_x += alpha[i] * f_values[i].dot(x)

    return f_x

def train_one_versus_all_logistic_classifier(features, labels, classifier = 'LR'):
    N = len(labels)
    n_labels = len(set(labels))
    alphas = np.zeros((n_labels, N))
    for label in range(n_labels):
        one_versus_all_labels = np.zeros(N)
        for i in range(N):
            if labels[i] == label:
                one_versus_all_labels[i] = 1
            else:
                one_versus_all_labels[i] = -1
        if classifier == 'LR':
            alphas[label, :] = logistic_regression_classifier(features, one_versus_all_labels)
        elif classifier == 'SVM':
            alphas[label, :] = SVM_classifier(features, one_versus_all_labels)
        print "classifier for label ", label, " done"
    return alphas

def test_one_versus_all_logistic_classifier(alphas, train_features, test_features, n_labels, test_labels=None, filename='../data/Yte.csv'):
    N = test_features.shape[0]
    predicted_labels = np.zeros(N, dtype=np.int)

    for i in range(N):
        test_feature = test_features[i,:]
        max_probability = 0
        for label in range(n_labels):
            p_label = sigma(predict(alphas[label,:], train_features, test_feature))
            if (p_label > max_probability):
                predicted_labels[i] = label
                max_probability = p_label

    if test_labels is None:
        print "writing result to file ", filename
        DF = pd.DataFrame(data=predicted_labels, columns=['Prediction'])
        DF.index += 1
        DF.to_csv(filename, index=True, index_label='Id', sep=',')
    else:
        error_rate = 0
        for c in predicted_labels:
            if (c != test_labels[i]):
                error_rate += 1.0
        print "error rate on test : ", error_rate / N

###########################
#ALL SVM RELATED FUNCTIONS#
###########################
def dual_objective_function(y,X,x):
    return sum(x)-0.5*np.linalg.norm(sum(np.multiply(x,np.multiply(y,X)).T))**2

def transform_svm_dual( C, X, y ):

    (d, n) = X.shape
    M = (y*X);
    Q = M.T.dot(M)
    p = -np.ones(n)
    b = np.zeros(2*n)
    b[:n] = C
    A = np.zeros((2*n , n))
    A[:n , :n] = np.eye(n)
    A[n:, :n] = -np.eye(n)

    return Q,p,A,b

def transform_svm_primal(C,X,y):

    [d, n] = X.shape

    Q = np.zeros((d+n, d+n))
    p = C*np.ones(d+n)
    for k in range(d):
        Q[k,k]=1
        p[k] = 0

    A = np.zeros((2*n, d+n))
    A[:n,:d] = np.multiply(y,X).T
    A[:n, d:] = np.eye(n)
    A[n:, d:] = np.eye(n)
    A=-A

    b = np.zeros(2*n)
    b[:n]=-np.ones(n)

    return Q, p, A, b

def f1(x, t):
    B=0
    for i in range(A.shape[0]):
        B = B -np.log(b[i]-A[i,:].dot(x))
    return t*0.5*(x.T).dot(Q.dot(x))+t*(p).dot(x)+B

def grad_f1(x, t):
    c = np.zeros(A.shape[0]);
    for i in range(len(c)):
        c[i] = 1/(b[i]-A[i,:].dot(x))

    return t*Q.T.dot(x) + t*p + A.T.dot(c)


def hess_f1(x, t):

    c = np.zeros(A.shape[0]);
    for i in range(len(c)):
        c[i] = 1/(b[i]-A[i,:].dot(x))

    M1 = np.zeros((A.shape[1], A.shape[1]))
    for i in range(len(c)):
        M1 = M1 + A[[i], :].T.dot(A[[i], :])/(b[i]-A[i,:].dot(x))

    return t*Q + M1


def Newton(t,x, grad_f1, hess_f1, f1):
    beta = 0.5
    alpha = 0.25

    delta_x = -np.linalg.inv(hess_f1(x,t)).dot(grad_f1(x, t))
    lambda1 = np.sqrt(grad_f1(x, t).T.dot(np.linalg.inv(hess_f1(x,t)).dot(grad_f1(x, t))))

    #Backtracking line search
    step = 1

    while (np.isnan(f1(x+step*delta_x, t))) or (f1(x+step*delta_x, t) >= f1(x, t)+alpha*step*grad_f1(x, t).T.dot(delta_x)):
        step = beta*step

    x_new = x+step*delta_x
    gap = lambda1**2/2
    #value_function.append(dual_objective_function(y,X,x_new))

    return x_new, gap

def centering_step(Q,p,A,b,x,t,tol):

    gap = np.inf

    while gap > tol:
        [x_new, gap] = Newton(t,x, grad_f1, hess_f1, f1)
        x = x_new

    return x

def barr_method(Q,p,A,b,x_0,mu,tol):

    x = x_0
    m = A.shape[0]
    t=1

    while m/t >= tol:
        x = centering_step(Q,p,A,b,x,t,tol)
        t = mu*t
    return x

def SVM_classifier(X, y, primal=True):
    C = 1
    tol = 0.5
    mu = 2
    t =1
    if primal:
        global Q, p, A, b
        Q, p, A, b = transform_svm_primal( C, X, y)
        d = X.shape[0]
        x_0 = np.zeros(A.shape[1])
        x_0[d:] = 2
    else:
        Q, p, A, b = transform_svm_dual( C, X, y)
        x_0 = 0.5*np.ones(A.shape[1])

    x = barr_method(Q,p,A,b,x_0,mu,tol)
    y_pred = X.T.dot(x[:d])

    return y_pred

    # faire une fonction qui permet de sortir x
