import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

INF = 8

def plot(
        signal, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    # set y range of 
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    # plt.show()

def init_signal():
    return np.zeros(2 * INF + 1)


def time_scale_signal(x : np.ndarray, k : int) -> np.ndarray:
    # implement this function
    t = np.arange(-INF, INF + 1)
    t_query = t/k 
    n = len(t)
    x_int = time_scale_signal_interpolate(x, k)
    invalid = (t_query < -INF) | (t_query > INF)
    return np.where(invalid, 0, x_int)


def time_scale_signal_interpolate(x : np.ndarray, k : int) -> np.ndarray:
    # implement this function
    t = np.arange(-INF, INF + 1)
    t_query = t/k 
    t0 = t[0]
    d = t[1]-t[0]
    n= len(t)

    idx_float = t_query-t0/ d 
    idx_round = np.round(idx_float)
    close = np.isclose(idx_float, idx_round, atol=1e-6)

    idx_left = np.where(close, idx_round, np.floor(idx_float)).astype(int)
    idx_right = np. where(close, idx_round, np.ceil(idx_float)).astype(int)

    idx_left = np.clip(idx_left,0,n-1)
    idx_right= np.clip(idx_right, 0 , n-1)

    new_x = 0.5 * (x[idx_left]+x[idx_right])
    return new_x




def main():
    img_root = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
    plot(time_scale_signal(signal, 3), title='x[n/3]', saveTo=f'{img_root}/x[n divided by 3].png')
    plot(time_scale_signal(signal, 1), title='x[n/1]', saveTo=f'{img_root}/x[n divided by 1].png')
    plot(time_scale_signal_interpolate(signal, 3), title='x[n/3] with interpolation', saveTo=f'{img_root}/x[n divided by 3]_with_interpolation.png')
    plot(time_scale_signal_interpolate(signal, 1), title='x[n/1] with interpolation', saveTo=f'{img_root}/x[n divided by 1]_with_interpolation.png')

main()
