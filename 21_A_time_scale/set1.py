import numpy as np
import matplotlib.pyplot as plt

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


def time_shift_signal(x : np.ndarray, k : int) -> np.ndarray:
    # implement this function
    if k > 0 : #this is delay 
        shifted = np.concatenate((np.zeros(abs(k)),x))[:len(x)]
    else : 
        shifted = np.concatenate((x[abs(k):], np.zeros(abs(k))))
    return shifted

def time_scale_signal(x : np.ndarray, k : int) -> np.ndarray:
    # implement this function
    n = np.arange(-INF, INF + 1 )
    idx = k * n + INF
    idx_clip = np.clip(idx,0,2*INF)
    valid = (idx < 0) | (idx >= 2*INF)
    return np.where(valid, 0, x[idx_clip])
    


def main():
    img_root_path = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root_path}/x[n].png')

    plot(time_shift_signal(signal, 2), title='x[n-2]', saveTo=f'{img_root_path}/x[n-2].png')
    
    plot(time_shift_signal(signal, -2), title='x[n+2]', saveTo=f'{img_root_path}/x[n+2].png')
    
    plot(time_shift_signal(signal, 0), title='x[n+0]', saveTo=f'{img_root_path}/x[n+0].png')
    
    plot(time_scale_signal(signal, 2), title='x[2n]', saveTo=f'{img_root_path}/x[2n].png')
    
    plot(time_scale_signal(signal, 1), title='x[1n]', saveTo=f'{img_root_path}/x[1n].png')
    
        

main()

#############solve
# def time_shift_signal(x : np.ndarray, k : int) -> np.ndarray:
#     # implement this function
#     return np.roll(x,k)
#     None

# def time_scale_signal(x : np.ndarray, k : int) -> np.ndarray:
#     # implement this function
#     temp=np.zeros_like(x)
#     second_half=np.array(x[8::k])
#     x=np.flip(x)
#     x=np.array(x[8+k::k])
#     x=np.flip(x)
#     temp[8:8+np.size(second_half)]+=second_half
#     temp[8-np.size(x):8]+=x
#     return temp
#     None   todays solve using np