import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def fit_quadratic_model(state_dim, states, targets):
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
        
        targets.append(target + np.random.randn()*0.1)

    return (states, targets, V_0, V_x, V_xx)

def get_accuracy(states, targets, V_0_hat, V_x_hat, V_xx_hat):
    total_error = 0.0

    for state, target in zip(states, targets):
        state = state.reshape((-1, 1))

        estimate = np.dot(V_x.T, state)
        estimate += np.dot(state.T, np.dot(V_xx, state))
        estimate += V_0

        total_error += (target - estimate) ** 2

    total_error = total_error / float(len(targets))
    return total_error

if __name__ == "__main__":
    for i in range(10):
        print("Accuracy of random estimate:")
        r_V_0 = np.random.randn(1, 1)
        r_V_x = np.random.randn(3, 1)
        r_V_xx = np.random.randn(3, 3)
        states, targets, V_0, V_x, V_xx = generate_random_data(dim=3, size=100)
        acc = get_accuracy(states, targets, r_V_0, r_V_x, r_V_xx) 
        print("Error of least squares estimate was {}".format(acc))

    for size in [1, 5, 10, 100, 500]:
        print("\n{} data points:".format(size))
        total_acc = 0.0

        for _ in range(10):
            states, targets, V_0, V_x, V_xx = generate_random_data(dim=3, size=size)
            V_0_hat, V_x_hat, V_xx_hat = fit_quadratic_model(3, states, targets)
            acc = get_accuracy(states, targets, V_0_hat, V_x_hat, V_xx_hat)
            total_acc += acc

        avg_acc = total_acc / 10.0
        print("Average error of least squares estimate was {}".format(avg_acc))
