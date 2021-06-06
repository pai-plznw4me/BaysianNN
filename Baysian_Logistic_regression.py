import numpy as np
import matplotlib.pyplot as plt


def gardient_descent(f, y, x, w):
    lr = 0.0001
    delta = 0.0000001

    # gradient descent 
    derivate = (f(y, x, w + delta) - f(y, x, w)) / delta

    # estimate new theta 
    return w + derivate * lr, derivate


def get_prior(w, mu=0, std=1):
    prior = 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * ((w - mu) / std) ** 2)
    return prior


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_loglikelihood(y, x, w):
    """
    Args:
        y, ndarray, shape 2d, (N, 1)
        x, ndarray, shape 2d, (N, n_features)
        w, ndarray, shape 2d, (n_features, 1)
    """
    return np.sum(np.log((sigmoid(x @ w) ** y) * ((1 - sigmoid(x @ w)) ** (1 - y))))


def psi(y, x, w):
    return np.log(get_prior(w)) + get_loglikelihood(y, x, w)


def gardient_psi(y, x, w):
    delta = 0.00000001

    # gradient descent
    derivate = (psi(y, x, w + delta) - psi(y, x, w)) / delta
    return derivate


def hessian_psi(y, x, w):
    delta = 0.00000001
    return (gardient_psi(y, x, w + delta) - gardient_psi(y, x, w)) / delta


def get_posterior(y, x, w_hat):
    h = -hessian_psi(y, x, w_hat)
    return np.random.normal(w_hat.reshape(-1), 1 / h.reshape(-1), 10000), h


if __name__ == '__main__':
    # test dataset
    np.random.seed(0)
    x = np.random.normal(0, 2, size=1000).reshape(-1, 1)
    noise = np.random.normal(0, 3, 1000).reshape(-1, 1)

    y = 3 * x + 2 + noise
    y = np.where(y > 0, 1, 0)
    y = y.reshape(-1, 1)

    init_w = np.random.normal(0, 1, 1).reshape(1, 1)

    ws = w = np.random.normal(0, 1, 1000)

    ws = []
    drvs = []
    psi_values = []
    w = init_w
    xs = range(1000)
    for i in range(1000):
        psi_value = psi(y, x, w)
        w, drv = gardient_descent(psi, y, x, w)

        ws.append(w.reshape(-1))
        drvs.append(drv.reshape(-1))
        psi_values.append(psi_value.reshape(-1))

fig, axes = plt.subplots(1, 2)
fig.set_size_inches((10, 10))
axes = axes.ravel()

axes[0].plot(xs, np.array(drvs)[:, 0].tolist())
axes[0].set_title('derivates')
axes[1].plot(xs, np.array(ws)[:, 0].tolist())
axes[1].set_title('ws')
plt.show()

zs = get_posterior(y, x, w)
zs = zs.reshape(-1)
plt.hist(zs)
plt.show()
