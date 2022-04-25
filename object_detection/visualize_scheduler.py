import matplotlib.pyplot as plt
import numpy as np


def scheduler(epoch, n_min, n_max, T):
    return n_min + (1 / 2) * (n_max - n_min) * (1 + np.cos(epoch / T * np.pi))


if __name__ == '__main__':
    n_min = 1e-9
    n_max = 1e-3
    Ts = [10, 25, 50, 60, 75, 100, 200]
    epochs = list(range(50))

    for T in Ts:
        values = []
        y_ticks = []
        for epoch in epochs:
            values.append(scheduler(epoch, n_min, n_max, T))
        plt.plot(epochs, values)
        print(T, min(values))

    plt.legend([str(x) for x in Ts])
    plt.show()
