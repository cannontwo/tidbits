import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.optimize import brentq

def get_intersections(x1, y1, x2, y2, limit_low, limit_high, depth, epsilon=0.1):
    """
    Get intersection points between the line segment between (x1, y1) and (x2,
    y2) and the curve y = sin(2x). Intersections are numerically and recursively.
    """
    intersections = []

    if depth > 9:
        return []

    def _func(x):
        return np.sin(2 * x) - ((y2 - y1) / float(x2 - x1)) * (x - x1) - y1

    if np.sign(_func(limit_low)) == np.sign(_func(limit_high)):
        # Brentq won't work in this case, so make arbitrary divisions
        print("Endpoint signs are the same, splitting")
        intersections.extend(get_intersections(x1, y1, x2, y2, limit_low, (limit_low + limit_high) / 2.0, depth+1))
        intersections.extend(get_intersections(x1, y1, x2, y2, (limit_low + limit_high) / 2.0, limit_high, depth+1))
        return intersections
    
    try:
        res = brentq(_func, limit_low, limit_high)
        print("Found root at {} in [{},{}]".format(res, limit_low, limit_high))
    except:
        print("Brentq max iterations exceeded")
        return []

    intersections.append(res)
    intersections.extend(get_intersections(x1, y1, x2, y2, limit_low, res - epsilon, depth+1))
    intersections.extend(get_intersections(x1, y1, x2, y2, res + epsilon, limit_high, depth+1))

    return intersections

if __name__ == "__main__":
    for _ in range(10):
        x1, y1, x2, y2 = np.random.uniform(-2, 2, size=4)
        limit_low = min(x1, x2)
        limit_high = max(x1, x2)
        print("x1, y1 are {}, {}".format(x1, y1))
        print("x2, y2 are {}, {}".format(x2, y2))
        start = time.time()
        print(get_intersections(x1, y1, x2, y2, limit_low, limit_high, 0))
        end = time.time()
        print("Took {} seconds".format(end - start))
        X = np.linspace(-2, 2, 100)
        Y = np.sin(2 * X)
        X2 = np.linspace(limit_low, limit_high, 100)
        Z = ((y2 - y1) / float(x2 - x1)) * (X2 - x1) + y1
        plt.close()
        plt.plot(X, Y)
        plt.plot(X2, Z)
        plt.show()
