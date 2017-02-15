from classification import *
import numpy as np

n_points = 50
points = np.random.rand(n_points, 2)
w = np.array([0.5, -0.5])

labels = np.zeros(n_points)

for i in range(0, n_points):
    labels[i] = -1 + 2 * int(w.dot(points[i,:]) > 0)

alpha = logistic_regression_classifier(points, labels)

print(alpha)

K = compute_K(points)
f_x = K.dot(alpha)

for i in range(0, n_points):
    print "true label ", labels[i], " pred label ", 2 * int(f_x[i] > 0) - 1
