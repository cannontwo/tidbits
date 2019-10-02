import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from sklearn import linear_model

def generate_quadratic_data(size=500):
    a = np.random.uniform(-10.0, 10.0)
    b = np.random.uniform(-10.0, 10.0)
    c = np.random.uniform(-10.0, 10.0)
    d = np.random.uniform(-10.0, 10.0)

    inputs = []
    data = []
    for _ in range(size):
        x = np.random.uniform(-1.0, 1.0)
        inputs.append(x)
        data.append(a * x ** 3 + b * x ** 2 + c * x + d)

    return (inputs, data)

if __name__ == "__main__":
    inputs, data = generate_quadratic_data()

    num_ref_points = 10
    ref_points = []
    ref_data = {}
    for i in range(num_ref_points):
        ref_points.append(np.array([np.random.uniform(-1.0, 1.0)]))
        ref_data[i] = []

    print(ref_points)

    kdt = KDTree(np.array(ref_points), leaf_size=2, metric='euclidean')
    for x, y in zip(inputs, data):
        ref_index = kdt.query(np.array([x]).reshape(1, -1), k=1, return_distance=False)[0][0]
        ref_data[ref_index].append(tuple([x, y]))

    print(ref_data)

    models = []
    for i in range(num_ref_points):
        ref = ref_points[i]
        orig_X = [x for x, y in ref_data[i]]
        X = [x - ref for x, y in ref_data[i]]
        Y = [y for x, y in ref_data[i]]
        print("X is {}".format(X))
        print("Y is {}".format(Y))
        clf = linear_model.Ridge(alpha=0.01)
        clf.fit(X, Y)
        models.append(clf)
        print(clf.predict(X))

        plt.plot([min(orig_X), max(orig_X)], [clf.predict([[min(orig_X) - ref[0]]]), clf.predict([[max(orig_X) - ref[0]]])])


    plt.scatter(inputs, data)
    plt.show()


