import numpy as np
import time

def compute_random_max():
    """
    Function to check that analytic max/min of quadratic from my research is
    correct. Running a bunch of Monte Carlo simulations is much easier than
    checking this symbolically, and also allows me to copy over the code.
    """

    A = np.random.randn(3, 3)
    B = np.random.randn(3, 2)
    c = np.random.randn(3, 1)

    V_x = np.random.randn(3, 1)
    V_xx = np.random.randn(3, 3)

    I = np.eye(2, 2)

    K = -1.0 * np.linalg.inv(I + B.T.dot(V_xx.dot(B))).dot(B.T.dot(V_xx.dot(A)))
    k = -1.0 * np.linalg.inv(I + B.T.dot(V_xx.dot(B))).dot((1.0/2.0) * B.T.dot(V_x) + B.T.dot(V_xx.dot(c)))

    s = np.random.randn(3, 1)

    print("K is {}".format(K))
    print("k is {}".format(k))

    inter1 = 2.0 * (K.dot(s) + k)
    inter2 = B.T.dot(V_x)

    inner = A.dot(s) + B.dot(K.dot(s) + k) + c
    inter3 = 2.0 * B.T.dot(V_xx).dot(inner)

    print("Total gradient is {}".format(inter1 + inter2 + inter3))
    return inter1 + inter2 + inter3

if __name__ == "__main__":
    num = 1000
    total_norm = 0.0
    max_norm = -float('inf')
    start = time.time()
    for _ in range(num):
        print("\n")
        grad = compute_random_max()
        total_norm += np.linalg.norm(grad)
        max_norm = max(max_norm, np.linalg.norm(grad))
    end = time.time()

    print("\nTime taken: {}".format(end - start))
    print("\tAverage time: {}".format((end - start) / float(num)))
    print("\nAverage gradient norm at fixed point is {}".format(total_norm / float(num)))
    print("\tMax gradient norm at fixed point is {}".format(max_norm))
