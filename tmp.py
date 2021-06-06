import numpy as np


def gardient(f, w):
    lr = 0.0001
    delta = 0.00000001

    # gradient descent
    derivate = (f(w + delta) - f(w)) / delta

    return derivate


def fn(x):
    return x ** 3


def hessian(f, w):
    delta = 0.00000001
    return (gardient(f, w + delta) - gardient(f, w)) / delta


if __name__ == '__main__':
    x = np.arange(0, 1, 0.1)
    print(hessian(fn, x))
