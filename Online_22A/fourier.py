import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Setup Time and Frequency Grids
# ==========================================
# Time domain parameters
T_max = 50.0            # Integration range [-T_max, T_max]
dt = 0.001              # Time step resolution
t = np.arange(-T_max, T_max, dt)

# Frequency domain parameters
f_max = 2.0             # Frequency range [-f_max, f_max] in Hz
df = 0.005              # Frequency step resolution
f = np.arange(-f_max, f_max, df)

# ==========================================
# 2. Define Signal & Analytical Derivatives
# ==========================================
x = 0.5 * np.cos(4 * t) + 0.5 * np.sin(6 * t)

# Analytical derivatives
y1 = -2.0 * np.sin(4 * t) + 3.0 * np.cos(6 * t)                     # 1st derivative
y2 = -8.0 * np.cos(4 * t) - 18.0 * np.sin(6 * t)                    # 2nd derivative
y3 = 32.0 * np.sin(4 * t) - 108.0 * np.cos(6 * t)                   # 3rd derivative

derivatives = [y1, y2, y3]

# ==========================================
# 3. Custom Numerical CFT Function (No FFT)
# ==========================================
def compute_cft(signal, t_grid, f_grid):
    """
    Computes Continuous Fourier Transform using trapezoidal integration.
    X(f) = integral( signal(t) * exp(-j * 2 * pi * f * t) dt )
    """
    X_f = np.zeros(len(f_grid), dtype=complex)
    
    # Loop over each frequency and integrate using np.trapz / np.trapezoid
    for i, freq in enumerate(f_grid):
        kernel = np.exp(-1j * 2 * np.pi * freq * t_grid)
        integrand = signal * kernel
        # Use np.trapezoid if NumPy >= 2.0, fallback to np.trapz
        if hasattr(np, 'trapezoid'):
            X_f[i] = np.trapezoid(integrand, t_grid)
        else:
            X_f[i] = np.trapz(integrand, t_grid)
            
    return X_f

# ==========================================
# 4. Compute Transforms and Verify Property
# ==========================================
# CFT of original signal x(t)
X_f = compute_cft(x, t, f)

# Storage for metrics and analysis
mse_mag_list = []
mse_phase_list = []

plt.figure(figsize=(14, 12))

for k in range(1, 4):
    # Method A: Theoretical Property -> Y_k(f) = (j * 2 * pi * f)^k * X(f)
    Y_prop = ((1j * 2 * np.pi * f) ** k) * X_f
    
    # Method B: Direct CFT of k-th derivative -> F{ d^k/dt^k x(t) }
    Y_direct = compute_cft(derivatives[k-1], t, f)
    
    # Extract Magnitude and Phase
    mag_prop = np.abs(Y_prop)
    mag_direct = np.abs(Y_direct)
    
    phase_prop = np.angle(Y_prop)
    phase_direct = np.angle(Y_direct)
    
    # Compute Mean Squared Error (MSE)
    mse_mag = np.mean((mag_direct - mag_prop) ** 2)
    mse_phase = np.mean((phase_direct - phase_prop) ** 2)
    
    mse_mag_list.append(mse_mag)
    mse_phase_list.append(mse_phase)
    
    # --- Plotting Results ---
    # Magnitude Comparison
    plt.subplot(3, 2, 2 * k - 1)
    plt.plot(f, mag_direct, 'b-', label=f'Direct CFT $|Y_{k}(f)|$')
    plt.plot(f, mag_prop, 'r--', label=f'Property $|(j2\pi f)^{k}X(f)|$')
    plt.title(f'Derivative {k}: Magnitude Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Phase Comparison
    plt.subplot(3, 2, 2 * k)
    plt.plot(f, phase_direct, 'b-', label=f'Direct Phase $\\angle Y_{k}(f)$')
    plt.plot(f, phase_prop, 'r--', label=f'Property Phase $\\angle (j2\pi f)^{k}X(f)$')
    plt.title(f'Derivative {k}: Phase Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 5. Print MSE Analysis & Comments
# ==========================================
print("=== MSE Analysis Summary ===")
for k in range(1, 4):
    print(f"Order {k} Derivative:")
    print(f"  - Magnitude MSE : {mse_mag_list[k-1]:.6e}")
    print(f"  - Phase MSE     : {mse_phase_list[k-1]:.6e}")

