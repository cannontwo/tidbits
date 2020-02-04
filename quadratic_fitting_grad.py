import sys

import numpy as np
import matplotlib.pyplot as plt

from math import factorial
from itertools import chain, combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures

class ADAMUpdater:
    """
    Class representing the state of an ADAM optimizer.
    """
    def __init__(self, dim):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8

        self.t = 0
        self.learning_rate = 0.001

        comb = int(factorial(dim) / factorial(2) / factorial(dim - 2))
        self.first_moments = np.zeros((1 + dim + dim ** 2 - comb, 1))
        self.second_moments = np.zeros((1 + dim + dim ** 2 - comb, 1))

    def apply_update(self, params, gradient):
        self.t += 1

        self.first_moments = self.beta_1 * self.first_moments + (1.0 - self.beta_1) * gradient
        self.second_moments = self.beta_2 * self.second_moments + (1.0 - self.beta_2) * np.power(gradient, 2)

        corr_first_moments = self.first_moments / (1.0 - (self.beta_1 ** self.t))
        corr_second_moments = self.second_moments / (1.0 - (self.beta_2 ** self.t))

        update = corr_first_moments / (np.power(corr_second_moments, 1.0/2.0) + self.epsilon)

        return params + self.learning_rate * update

def generate_data(batch_size, dim=1, noise_scale=1.0):
    print("Generating {} datapoints".format(batch_size))

    V_0 = np.random.randn(1, 1)
    V_x = np.random.randn(dim, 1)
    V_xx = np.random.randn(dim, dim)

    inputs = np.random.randn(batch_size, dim)
    total_error = 0.0
    targets = []

    for x in inputs:
        xr = x.reshape((-1, 1))
        noise_term = noise_scale * np.random.randn(1, 1)
        total_error += noise_term

        targets.append(V_0 + V_x.transpose().dot(xr) + xr.transpose().dot(V_xx).dot(xr) + noise_term)

    targets = np.array(targets).reshape((-1, 1))

    return V_0, V_x, V_xx, inputs, targets, total_error

def compute_ls_grad(X, Y, params):
    denom = float(len(Y))
    #total_grad = np.zeros_like(params)

    #for i in range(len(Y)):
    #    x_row = X[i].reshape((1, -1))
    #    partial_grad = x_row.transpose().dot(x_row).dot(params)
    #    total_grad += partial_grad - x_row.transpose() * Y[i]
    grad = X.transpose().dot(X).dot(params) - X.transpose().dot(Y)

    return -grad / denom

def do_gradient_descent(inputs, targets, num_iters, mini_size, dim=1):
    """
    Fit a quadratic approximation to the input data using ADAM gradient descent.
    """
    opt = ADAMUpdater(dim)
    poly = PolynomialFeatures(2)

    X = poly.fit_transform(inputs)
    comb = int(factorial(dim) / factorial(2) / factorial(dim - 2))
    params = np.zeros((1 + dim + dim ** 2 - comb, 1))

    error = []
    print(X.dot(params).shape, targets.shape)
    assert(X.dot(params).shape == targets.shape)

    for j in range(num_iters):
        i = 0
        print("On iteration {}".format(j))
        while i < len(inputs):
            mini_inputs = X[i:i+mini_size]
            mini_targets = targets[i:i+mini_size]
            i += mini_size
            grad = compute_ls_grad(mini_inputs, mini_targets, params) 
            params = opt.apply_update(params, grad)
            error.append(np.linalg.norm(X.dot(params) - targets))

    powers = list(chain.from_iterable(combinations_with_replacement(range(dim), i) for i in range(3)))

    V_0 = params[0].reshape((1, 1))
    V_x = params[1:dim+1].reshape((dim, 1))
    V_xx = np.zeros((dim, dim))

    for idx, power in enumerate(powers):
        if len(power) == 2:
            assert(idx >= dim+1)
            V_xx[power[0], power[1]] += params[idx] / 2.0
            V_xx[power[1], power[0]] += params[idx] / 2.0

    return V_0, V_x, V_xx, error

def run(num_iters, batch_size=1000, mini_size=64, dim=20):
    """
    Do 'num_iters' iterations of ADAM optimization with given batch size and
    minibatch size. Data generated from an underlying quadratic model with
    additive Gaussian noise.
    """
    print("Running {} iterations of ADAM with a batch_size of {} and minibatch size of {}".format(num_iters, batch_size, mini_size))

    V_0, V_x, V_xx, inputs, targets, real_error = generate_data(batch_size, dim=dim)
    
    #xs = np.linspace(-5, 5).reshape((-1, 1))
    #ys = (V_0 + V_x.dot(xs.transpose()) + np.diag(xs.dot(V_xx).dot(xs.transpose()))).reshape(-1, 1)

    V_0_hat, V_x_hat, V_xx_hat, error = do_gradient_descent(inputs, targets, num_iters, mini_size, dim=dim)

    #ys_hat = (V_0_hat + V_x_hat.dot(xs.transpose()) + np.diag(xs.dot(V_xx_hat).dot(xs.transpose()))).reshape(-1, 1)

    #plt.scatter(inputs, targets)
    #plt.plot(xs, ys, 'm')
    #plt.plot(xs, ys_hat, 'g')
    #plt.legend(['true', 'grad_fit', 'data'])

    plt.figure()
    plt.plot(error)
    plt.hlines(real_error, 0, len(error), linestyles=['dashed'])
    plt.show() 


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage: {} NUM_ITERATIONS [BATCH_SIZE] [MINIBATCH_SIZE]".format(sys.argv[0]))

    if len(sys.argv) == 2:
        run(int(sys.argv[1]))
    elif len(sys.argv) == 3:
        run(int(sys.argv[1]), batch_size=int(sys.argv[2]))
    else:
        run(int(sys.argv[1]), batch_size=int(sys.argv[2]), mini_size=int(sys.argv[3]))


