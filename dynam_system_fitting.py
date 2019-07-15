import numpy as np

from itertools import chain, combinations_with_replacement

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def fit_normal(state_dim, action_dim, data): 
    # Use normal equations to solve least squares estimation problem
    A = np.zeros((state_dim, state_dim))
    B = np.zeros((state_dim, action_dim))
    c = np.zeros((state_dim, 1))
    theta = np.concatenate([A.T, B.T, c.T])

    for trans in data:
        state, action, state_prime = trans
        phi = np.concatenate([state.reshape(-1,1).T, action.reshape(-1,1).T, np.ones((1, 1)).T], axis=1)
        theta_hat = np.dot(np.linalg.pinv(phi), state_prime.reshape(-1,1).T)
        theta += theta_hat

    theta /= float(len(data))

    A = theta[:state_dim, :state_dim].T
    B = theta[state_dim:-1, :].T
    c = theta[-1:, :].T.reshape((state_dim,1))

    return (A, B, c)

def sk_fit_model(data):
    X = None
    Y = None

    for state, action, new_state in data:
        feat = np.concatenate([state, action])

        if X is None:
            X = feat.reshape((1, -1))
        else:
            X = np.concatenate([X, feat.reshape((1, -1))])
        
        if Y is None:
            Y = new_state.reshape((1, -1))
        else:
            Y = np.concatenate([Y, new_state.reshape((1, -1))])

    clf = linear_model.LinearRegression()
    clf.fit(X, Y)

    return clf

def generate_random_data(state_dim, action_dim, size=100):
    A = np.random.randn(state_dim, state_dim)
    B = np.random.randn(state_dim, action_dim)
    c = np.random.randn(state_dim, 1)

    data = []
    for i in range(size):
        state = np.random.randn(state_dim, 1)
        action = np.random.randn(action_dim, 1)
        new_state = np.dot(A, state) + np.dot(B, action) + c

        data.append((state, action, new_state))

    return (data, A, B, c)

def generate_more_data(A, B, c, state_dim, action_dim):
    data = []
    for i in range(size):
        state = np.random.randn(state_dim, 1)
        action = np.random.randn(action_dim, 1)
        new_state = np.dot(A, state) + np.dot(B, action) + c

        data.append((state, action, new_state))

    return data

def get_accuracy(data, A, B, c):
    total_error = 0.0

    for state, action, next_state in data:
        estimate = np.dot(A, state)
        estimate += np.dot(B, action)
        estimate += c

        total_error += np.linalg.norm(estimate - next_state)

    avg_error = total_error / float(len(data))
    return avg_error

def sk_get_accuracy(data, model):
    total_error = 0.0

    for state, action, new_state in data:
        feat = np.concatenate([state, action])

        estimate = model.predict(feat.reshape((1, -1)))

        total_error += np.linalg.norm(new_state - estimate.reshape((-1, 1)))

    avg_error = total_error / float(len(data))
    return avg_error

def get_mats_from_sk(model, state_dim, action_dim):
    coefs = model.coef_

    c = model.intercept_.reshape((-1, 1))
    A = coefs[:, :state_dim]
    B = coefs[:, state_dim:]

    return A, B, c

if __name__ == "__main__":
    state_dim = 6
    action_dim = 2
    for i in range(10):
        print("Accuracy of random estimate:")
        r_c = np.random.randn(state_dim, 1)
        r_A = np.random.randn(state_dim, state_dim)
        r_B = np.random.randn(state_dim, action_dim)

        data, A, B, c = generate_random_data(state_dim, action_dim, size=100)
        acc = get_accuracy(data, r_A, r_B, r_c) 
        print("Error of random estimate was {}".format(acc))

    for size in [1, 5, 10, 100, 500]:
        print("\n{} data points:".format(size))
        total_acc = 0.0
        sk_total_acc = 0.0
        reconstructed_sk_total_acc = 0.0

        for _ in range(10):
            data, A, B, c = generate_random_data(state_dim, action_dim, size=size)

            A_hat, B_hat, c_hat = fit_normal(state_dim, action_dim, data)

            sk_model = sk_fit_model(data)
            
            more_data = generate_more_data(A, B, c, state_dim, action_dim)

            acc = get_accuracy(more_data, A_hat, B_hat, c_hat)
            total_acc += acc
            
            sk_acc = sk_get_accuracy(more_data, sk_model)
            sk_total_acc += sk_acc

            sk_A, sk_B, sk_c = get_mats_from_sk(sk_model, state_dim, action_dim)

            reconstructed_acc = get_accuracy(more_data, sk_A, sk_B, sk_c)
            reconstructed_sk_total_acc += reconstructed_acc

        avg_acc = total_acc / 10.0
        avg_sk_acc = sk_total_acc / 10.0
        avg_reconstructed_acc = reconstructed_sk_total_acc / 10.0

        print("Average error of least squares estimate was {}".format(avg_acc))
        print("Average error of sklearn estimate was {}".format(avg_sk_acc))
        print("Average reconstructed error of sklearn estimate was {}".format(avg_reconstructed_acc))
