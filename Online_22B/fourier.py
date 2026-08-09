import numpy as np
import matplotlib.pyplot as plt

class CFTHelper:

    def compute_cft(self, signal, t_grid, f_grid):
        
        signal = np.asarray(signal)
        t_grid = np.asarray(t_grid)
        f_grid = np.asarray(f_grid)

        X_f = np.zeros(len(f_grid), dtype=complex)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        for i, freq in enumerate(f_grid):
            kernel = np.exp(-1j * 2 * np.pi * freq * t_grid)
            X_f[i] = integrate(signal * kernel, t_grid)

        return X_f

 
    def mse_report_custom(self, signal_a, signal_b, t0,freq,label="Comparison", verbose=True):
        
        a = np.asarray(signal_a)
        b = np.asarray(signal_b)

        mse_mag = np.mean((np.abs(a) - np.abs(b)) ** 2)
        phase_diff = np.angle(a) - (2 * np.pi * freq * t0) - np.angle(b)
        phase_diff = np.angle(np.exp(1j * phase_diff))
        mse_phase = np.mean(phase_diff ** 2)
        
        if verbose:
            print(f"[{label}] MSE magnitude: {mse_mag:.6e} | MSE phase: {mse_phase:.6e}")

        return {"label": label, "mse_mag": mse_mag, "mse_phase": mse_phase}


    def plot_comparison(self, pairs, f_grid, titles=None, labels=("Direct", "Property"), figsize=None):
        
        k = len(pairs)
        if titles is None:
            titles = [f"Signal {i + 1}" for i in range(k)]
        if figsize is None:
            figsize = (14, 4 * k)

        fig = plt.figure(figsize=figsize)

        for i, (sig_a, sig_b) in enumerate(pairs):
            sig_a = np.asarray(sig_a)
            sig_b = np.asarray(sig_b)

            # Magnitude
            plt.subplot(k, 2, 2 * i + 1)
            plt.plot(f_grid, np.abs(sig_a), 'b-', label=f'{labels[0]} $|{"{"}{titles[i]}{"}"}|$')
            plt.plot(f_grid, np.abs(sig_b), 'r--', label=f'{labels[1]}')
            plt.title(f'{titles[i]}: Magnitude Comparison')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid(True)

            # Phase
            plt.subplot(k, 2, 2 * i + 2)
            plt.plot(f_grid, np.angle(sig_a), 'b-', label=f'{labels[0]} Phase')
            plt.plot(f_grid, np.angle(sig_b), 'r--', label=f'{labels[1]} Phase')
            plt.title(f'{titles[i]}: Phase Comparison')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase (radians)')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        return fig


T_MAX = 5
F_MAX = 10
t0 = 1

dt = 10/2000
t = np.arange(-T_MAX, T_MAX,dt )
f= np.arange(-F_MAX, F_MAX, 20/1000)
t_2 = t * t 
x_t = np.exp(-1*t_2)
t_y = t + t0
y_t = x_t.copy()





helper = CFTHelper()
X_f = helper.compute_cft(x_t, t, f)
Y_f = helper.compute_cft(y_t, t_y, f)

pairs=[(X_f, Y_f)]
helper.plot_comparison(pairs,f,"Magnitude and phase comparison",("X_F","Y_F"))

helper.mse_report_custom(X_f, Y_f, t0, f,"Comparison",True)
plt.show()




