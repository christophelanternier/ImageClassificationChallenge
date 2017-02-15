import cvxopt
import numpy as np

def logistic_loss(f_x, y):
    return np.log(1 + np.exp(-y * f_x))

def logistic_loss_prime(f_x, y):
    return - 1.0 / (1 + np.exp(y * f_x))

def logistic_regression_loss_function(alpha, f_values, labels, _lambda):
    N = len(alpha)
    K = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            K[i, j] = f_values[i,:].dot(f_values[j,:])
    f_x = K.dot(alpha)
    result = 0

    for i in range(0,N):
        f_x_i =f_x[i]
        y_i = labels[i]
        result += logistic_loss(f_x_i, y_i)

    result = result / N
    result += _lambda * alpha.dot(f_x)

    return result

def grad_logistic_regression_loss_function(alpha, f_values, labels, _lambda):
    N = len(alpha)
    K = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            K[i, j] = f_values[i,:].dot(f_values[j,:])
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

    default_step = 10000
    alpha_0 = np.zeros(N)
    stop_criterion = 10**(-3)
    gradient_norm = stop_criterion + 1
    n_iteration_max = 500
    iter = 0

    while (iter < n_iteration_max) and (gradient_norm > stop_criterion):
        step = default_step
        L = logistic_regression_loss_function(alpha_0, f_values, labels, _lambda)
        grad_L = grad_logistic_regression_loss_function(alpha_0, f_values, labels, _lambda)
        gradient_norm = np.linalg.norm(grad_L)
        L_update = logistic_regression_loss_function(alpha_0 - step * grad_L, f_values, labels, _lambda)
        while (L_update > L):
            step *= 0.5
            L_update = logistic_regression_loss_function(alpha_0 - step * grad_L, f_values, labels, _lambda)

        alpha_0 = alpha_0 - step * grad_L
        iter +=1

        print "iter ", iter, " alpha ", alpha_0

    return alpha_0
