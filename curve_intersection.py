import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.optimize import brentq

def get_intersections(x1, y1, x2, y2, limit_low, limit_high, depth, epsilon=0.01):
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
        intersections.extend(get_intersections(x1, y1, x2, y2, limit_low, (limit_low + limit_high) / 2.0, depth+1))
        intersections.extend(get_intersections(x1, y1, x2, y2, (limit_low + limit_high) / 2.0, limit_high, depth+1))
        return intersections
    
    try:
        res = brentq(_func, limit_low, limit_high)
        print("Found root at {} in [{},{}]".format(res, limit_low, limit_high))
    except:
        return []

    intersections.append(res)
    intersections.extend(get_intersections(x1, y1, x2, y2, limit_low, res - epsilon, depth+1))
    intersections.extend(get_intersections(x1, y1, x2, y2, res + epsilon, limit_high, depth+1))

    return intersections

def compute_seg_integral(x1, y1, x2, y2):
    # Here we will assume that the two curves are in the same orientation
    # throughout the whole curve that they are halfway through (to handle cases
    # where one or both of the endpoints are intersection points). This means
    # that this method is only mathematically valid when the end points
    # provided as arguments are coming from chopping up the original line
    # segment.
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0

    if mid_y > np.sin(2 * mid_x):
        # Line segment is above sine curve
        print("Line segment above sine")
        m = (y2 - y1) / (x2 - x1)
        integral = (x2 ** 2 / 2.0) - x2 * x1
        integral -= (x1 ** 2 / 2.0) - x1 * x1
        integral *= m
        integral += x2 * y1
        integral -= x1 * y1
        sin_part = (-1.0 / 2.0) * np.cos(2 * x2)
        sin_part += (1.0 / 2.0) * np.cos(2 * x1)
        integral -= sin_part
    else:
        print("Sine above line segment")
        # Sine curve is above line segment
        integral = (-1.0 / 2.0) * np.cos(2 * x2)
        integral += (1.0 / 2.0) * np.cos(2 * x1)
        m = (y2 - y1) / (x2 - x1)
        second_part = (x2 ** 2 / 2.0) - x2 * x1
        second_part -= (x1 ** 2 / 2.0) - x1 * x1
        integral -= m * second_part
        integral -= x2 * y1
        integral += x1 * y1

    return integral

def compute_full_integral(x1, y1, x2, y2, intersections, limit_low, limit_high):
    total = 0.0

    def _func(x):
        return ((y2 - y1) / float(x2 - x1)) * (x - x1) + y1
    
    s_intersections = sorted(intersections)
    s_intersections.insert(0, limit_low)
    s_intersections.append(limit_high)

    for i in range(len(s_intersections) - 1):
        x1_hat = s_intersections[i]
        x2_hat = s_intersections[i+1]
        y1_hat = _func(x1_hat)
        y2_hat = _func(x2_hat)

        print("Computing integral for segment on [{},{}]".format(x1_hat, x2_hat))
        part = compute_seg_integral(x1_hat, y1_hat, x2_hat, y2_hat)
        print("Line segment integral is {}".format(part))
        assert(part >= 0.0)
        total += part

    return total

if __name__ == "__main__":
    for i in range(10):
        x1, y1, x2, y2 = np.random.uniform(-2, 2, size=4)
        x1, y1 = -2.0, 0.0
        theta = (np.pi / 10) * i - np.pi / 2.0
        x2, y2 = 2.0 * np.cos(theta), 2.0 * np.sin(theta) 
        limit_low = min(x1, x2)
        limit_high = max(x1, x2)
        print("x1, y1 are {}, {}".format(x1, y1))
        print("x2, y2 are {}, {}".format(x2, y2))
        start = time.time()
        intersections = get_intersections(x1, y1, x2, y2, limit_low, limit_high, 0)
        print("Intersections points are {}".format(sorted(intersections)))
        end = time.time()
        print("Took {} seconds to compute intersections".format(end - start))

        start = time.time()
        print("Integral is {}".format(compute_full_integral(x1, y1, x2, y2, intersections, limit_low, limit_high)))
        end = time.time()
        print("Took {} seconds to compute integral\n".format(end - start))
        
        X = np.linspace(-2, 2, 100)
        Y = np.sin(2 * X)
        X2 = np.linspace(limit_low, limit_high, 100)
        Z = ((y2 - y1) / float(x2 - x1)) * (X2 - x1) + y1
        plt.close()
        plt.plot(X, Y)
        plt.plot(X2, Z)
        plt.show()
