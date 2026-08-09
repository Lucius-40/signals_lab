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

 
    def mse_report(self, signal_a, signal_b, label="Comparison", verbose=True):
        
        a = np.asarray(signal_a)
        b = np.asarray(signal_b)

        mse_mag = np.mean((np.abs(a) - np.abs(b)) ** 2)
        mse_phase = np.mean((np.angle(a) - np.angle(b)) ** 2)

        if verbose:
            print(f"[{label}] MSE magnitude: {mse_mag:.6e} | MSE phase: {mse_phase:.6e}")

        return {"label": label, "mse_mag": mse_mag, "mse_phase": mse_phase}

    # ----------------------------------------------------------------
    # 3. Plot magnitude/phase comparisons for k signal pairs
    # ----------------------------------------------------------------
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

if __name__ == "__main__":
    T_max, dt = 50.0, 0.001
    t = np.arange(-T_max, T_max, dt)
    f_max, df = 2.0, 0.005
    f = np.arange(-f_max, f_max, df)

    x = 0.5 * np.cos(4 * t) + 0.5 * np.sin(6 * t)
    y1 = -2.0 * np.sin(4 * t) + 3.0 * np.cos(6 * t)
    y2 = -8.0 * np.cos(4 * t) - 18.0 * np.sin(6 * t)
    y3 = 32.0 * np.sin(4 * t) - 108.0 * np.cos(6 * t)
    derivatives = [y1, y2, y3]

    helper = CFTHelper()
    X_f = helper.compute_cft(x, t, f)

    pairs = []
    for kk in range(1, 4):
        Y_prop = ((1j * 2 * np.pi * f) ** kk) * X_f
        Y_direct = helper.compute_cft(derivatives[kk - 1], t, f)
        helper.mse_report(Y_direct, Y_prop, label=f"Order {kk} Derivative")
        pairs.append((Y_direct, Y_prop))

    helper.plot_comparison(pairs, f, titles=[f"Derivative {i}" for i in range(1, 4)])
    plt.show()