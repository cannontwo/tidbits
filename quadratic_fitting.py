import numpy as np

from itertools import chain, combinations_with_replacement

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def fit_quadratic_model(state_dim, states, targets):
    """
    Fit quadratic model using least squares. Here, we're using the normal
    equations to solve for parameters arranged in a quadratic approximation of
    the form est = V_0 + V_x * state + state.T * V_xx * state.
    """

    V_0 = np.zeros((1, 1))
    V_x = np.zeros((state_dim, 1))
    V_xx = np.zeros((state_dim, state_dim))
    
    theta = np.concatenate([V_x.T, V_xx.flatten().reshape((-1, 1)).T, V_0.T], axis=1)

    for state, target in zip(states, targets):
        assert(len(state) == state_dim)

        quad_state = np.dot(state.reshape((state_dim,1)), state.reshape((state_dim,1)).T)

        phi = np.concatenate([state.reshape((-1, 1)), quad_state.flatten().reshape((-1, 1)), np.array([[1.0]])])
        theta_hat = np.linalg.pinv(phi) * target

        theta += theta_hat

    theta /= float(len(targets))

    for state, target in zip(states, targets):
        quad_state = np.dot(state.reshape((state_dim,1)), state.reshape((state_dim,1)).T)
        phi = np.concatenate([state.reshape((-1, 1)), quad_state.flatten().reshape((-1, 1)), np.array([[1.0]])])
        error = (target - np.dot(theta, phi)) ** 2

    
    V_x = theta[0, :state_dim].copy().reshape((state_dim, 1))
    V_xx = theta[0, state_dim:-1].reshape((state_dim, state_dim)).copy()
    V_0 = theta[0, -1].copy()

    return (V_0, V_x, V_xx)

def sk_fit_quadratic_model(states, targets):
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(states)

    clf = linear_model.LinearRegression()
    clf.fit(X_, targets)

    return clf

def generate_random_data(dim=3, size=50):
    V_0 = np.random.randn(1, 1)
    V_x = np.random.randn(dim, 1)
    V_xx = np.random.randn(dim, dim)

    states = np.random.randn(size, dim)

    targets = []
    for state in states:
        state = state.reshape((dim, 1))

        target = np.dot(V_x.T, state)
        target += np.dot(state.T, np.dot(V_xx, state))
        target += V_0
        
        targets.append(target)

    return (states, targets, V_0, V_x, V_xx)

def generate_more_data(V_0, V_x, V_xx, dim=3):
    states = np.random.randn(size, dim)

    targets = []
    for state in states:
        state = state.reshape((dim, 1))

        target = np.dot(V_x.T, state)
        target += np.dot(state.T, np.dot(V_xx, state))
        target += V_0
        
        targets.append(target)

    return (states, targets)

def get_accuracy(states, targets, V_0_hat, V_x_hat, V_xx_hat):
    total_error = 0.0

    for state, target in zip(states, targets):
        state = state.reshape((-1, 1))

        estimate = np.dot(V_x_hat.T, state)
        estimate += np.dot(state.T, np.dot(V_xx_hat, state))
        estimate += V_0_hat

        total_error += (target - estimate) ** 2

    avg_error = total_error / float(len(targets))
    return avg_error

def sk_get_accuracy(states, targets, model):
    total_error = 0.0

    for state, target in zip(states, targets):
        state = state.reshape((1, -1))
        poly = PolynomialFeatures(degree=2)

        estimate = model.predict(poly.fit_transform(state))

        total_error += (target - estimate) ** 2

    avg_error = total_error / float(len(targets))
    return avg_error

def get_mats_from_sk(model, dim=3):
    # This is the ordering that PolynomialFeatures uses
    powers = list(chain.from_iterable(combinations_with_replacement(range(dim), i) for i in range(3)))
    coefs = model.coef_

    print(coefs)
    print(model.intercept_)

    V_0 = np.zeros((1, 1))
    V_x = np.zeros((dim, 1))
    V_xx = np.zeros((dim, dim))

    for idx, power in enumerate(powers):
        if len(power) == 0:
            V_0 = model.intercept_
        elif len(power) == 1:
            V_x[power] += coefs[idx]
        elif len(power) == 2:
            V_xx[power[0], power[1]] += coefs[idx] / 2.0
            V_xx[power[1], power[0]] += coefs[idx] / 2.0

    return (V_0, V_x, V_xx)

if __name__ == "__main__":
    for i in range(10):
        print("Accuracy of random estimate:")
        r_V_0 = np.random.randn(1, 1)
        r_V_x = np.random.randn(3, 1)
        r_V_xx = np.random.randn(3, 3)
        states, targets, V_0, V_x, V_xx = generate_random_data(dim=3, size=100)
        acc = get_accuracy(states, targets, r_V_0, r_V_x, r_V_xx) 
        print("Error of random estimate was {}".format(acc))

    for size in [1, 5, 10, 100, 500]:
        print("\n{} data points:".format(size))
        total_acc = 0.0
        sk_total_acc = 0.0
        reconstructed_sk_total_acc = 0.0

        for _ in range(10):
            states, targets, V_0, V_x, V_xx = generate_random_data(dim=3, size=size)

            V_0_hat, V_x_hat, V_xx_hat = fit_quadratic_model(3, states, targets)
            sk_model = sk_fit_quadratic_model(states, np.array(targets).flatten())
            
            more_states, more_targets = generate_more_data(V_0, V_x, V_xx, dim=3)

            acc = get_accuracy(more_states, more_targets, V_0_hat, V_x_hat, V_xx_hat)
            total_acc += acc
            
            sk_acc = sk_get_accuracy(more_states, more_targets, sk_model)
            sk_total_acc += sk_acc

            sk_V_0, sk_V_x, sk_V_xx = get_mats_from_sk(sk_model, dim=3)

            reconstructed_acc = get_accuracy(more_states, more_targets, sk_V_0, sk_V_x, sk_V_xx)
            reconstructed_sk_total_acc += reconstructed_acc

        avg_acc = total_acc / 10.0
        avg_sk_acc = sk_total_acc / 10.0
        avg_reconstructed_acc = reconstructed_sk_total_acc / 10.0

        print("Average error of least squares estimate was {}".format(avg_acc))
        print("Average error of sklearn estimate was {}".format(avg_sk_acc))
        print("Average reconstructed error of sklearn estimate was {}".format(avg_reconstructed_acc))
