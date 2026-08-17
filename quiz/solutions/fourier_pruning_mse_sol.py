import numpy as np
import sys


# =====================================================================
# Given class — paste your offline implementation where indicated
# =====================================================================

class FourierEpicycles:
    """
    Represents a closed curve via its Fourier series coefficients.
    (Given — paste your offline implementation of compute_coeffs/reconstruct/
    evaluate_reconstruction_error/prune_harmonics_by_energy here.)
    """

    def __init__(self, t, f_vals, N, T):
        self.t = t                # sample times, shape (M,)
        self.f_vals = f_vals      # ground-truth complex signal x(t)+i*y(t), shape (M,)
        self.N = N                # harmonics range -N..N
        self.T = T                # period
        self.coeffs = None        # shape (2N+1,), index n+N -> c_n

    def compute_coeffs(self):
        """
        c_n = (1/T) * integral_0^T f(t) * exp(-i*2*pi*n*t/T) dt
        approximated as a Riemann sum over the M uniformly-spaced samples
        (dt = T / M, since t is endpoint=False over [0, T)).
        """
        M = len(self.t)
        dt = self.T / M
        ns = np.arange(-self.N, self.N + 1)
        # E[n_idx, m] = exp(-i*2*pi*n*t_m/T)   shape (2N+1, M)
        E = np.exp(-1j * 2 * np.pi * np.outer(ns, self.t) / self.T)
        coeffs = (E @ self.f_vals) * dt / self.T
        return coeffs

    def reconstruct(self, coeffs=None):
        """Rebuild f_hat(t) = sum_n c_n * exp(i*2*pi*n*t/T) at self.t."""
        if coeffs is None:
            coeffs = self.coeffs
        ns = np.arange(-self.N, self.N + 1)
        # E[m, n_idx] = exp(i*2*pi*n*t_m/T)   shape (M, 2N+1)
        E = np.exp(1j * 2 * np.pi * np.outer(self.t, ns) / self.T)
        return E @ coeffs

    def evaluate_reconstruction_error(self, coeffs=None):
        """Mean squared error between ground truth f_vals and the reconstruction."""
        f_hat = self.reconstruct(coeffs)
        return np.mean(np.abs(self.f_vals - f_hat) ** 2)

    def prune_harmonics_by_energy(self, r):
        """
        Keep the smallest set of harmonics (ranked by |c_n|^2, largest first)
        whose cumulative energy reaches a fraction r of the total energy.
        Zero out everything else.
        """
        energy = np.abs(self.coeffs) ** 2
        total_energy = np.sum(energy)
        order = np.argsort(energy)[::-1]
        cumulative = np.cumsum(energy[order])
        cutoff = np.searchsorted(cumulative, r * total_energy) + 1
        keep_idx = order[:cutoff]

        pruned = np.zeros_like(self.coeffs)
        pruned[keep_idx] = self.coeffs[keep_idx]
        return pruned


# =====================================================================
# Task A — Top-K magnitude pruning
# =====================================================================

class TopKPruner:
    """Practice: prune to the K harmonics with largest |c_n|, instead of by energy ratio."""

    def prune_top_k_magnitude(self, epicycles: FourierEpicycles, k):
        coeffs = epicycles.coeffs
        magnitudes = np.abs(coeffs)

        # Rank harmonics by |c_n| descending, keep top k
        order = np.argsort(magnitudes)[::-1]
        keep_array_idx = order[:k]

        pruned = np.zeros_like(coeffs)
        pruned[keep_array_idx] = coeffs[keep_array_idx]

        # convert array indices -> harmonic numbers n (index n+N -> n)
        retained_indices = sorted((keep_array_idx - epicycles.N).tolist())
        return pruned, retained_indices


# =====================================================================
# Task B — MSE vs N sweep
# =====================================================================

class MSESweep:
    """Practice: sweep reconstruction MSE as the number of retained harmonics grows."""

    def mse_vs_n_sweep(self, epicycles: FourierEpicycles, n_values):
        results = {}
        harmonic_numbers = np.arange(-epicycles.N, epicycles.N + 1)
        for n in n_values:
            mask = np.abs(harmonic_numbers) <= n
            truncated_coeffs = np.where(mask, epicycles.coeffs, 0)
            results[n] = epicycles.evaluate_reconstruction_error(truncated_coeffs)
        return results

    def mse_vs_k_sweep(self, epicycles: FourierEpicycles, pruner: TopKPruner, k_values):
        results = {}
        for k in k_values:
            pruned_coeffs, _ = pruner.prune_top_k_magnitude(epicycles, k)
            results[k] = epicycles.evaluate_reconstruction_error(pruned_coeffs)
        return results


# =====================================================================
# Validator — checks your filled-in methods against independent ground truth
# =====================================================================

class PruningValidator:
    """
    Runs independent checks against your implementations above.
    Does NOT reimplement your reconstruct()/MSE math — it checks the
    invariants those methods are supposed to satisfy.
    """

    def check_top_k_count_and_zeros(self, pruner: TopKPruner, epicycles: FourierEpicycles, k):
        pruned, retained = pruner.prune_top_k_magnitude(epicycles, k)
        nonzero_count = np.count_nonzero(pruned)
        count_ok = nonzero_count <= k
        # every retained coefficient should be among the k largest magnitudes
        order = np.argsort(np.abs(epicycles.coeffs))[::-1]
        true_top_k_indices = set(order[:k].tolist())
        kept_indices = set(np.nonzero(pruned)[0].tolist())
        subset_ok = kept_indices.issubset(true_top_k_indices)
        return (count_ok and subset_ok), nonzero_count

    def check_full_reconstruction_zero_error(self, epicycles: FourierEpicycles, tol=1e-6):
        """Using all coefficients (no pruning) should give ~0 MSE against ground truth."""
        mse = epicycles.evaluate_reconstruction_error(epicycles.coeffs)
        return mse < tol, mse

    def check_mse_monotonic_nonincreasing(self, mse_dict):
        """
        As the retained-harmonic budget (n or k) grows, MSE should not increase.
        mse_dict: {budget: mse}, ordered by ascending budget.
        """
        budgets = sorted(mse_dict.keys())
        mses = [mse_dict[b] for b in budgets]
        diffs = np.diff(mses)
        is_monotonic = np.all(diffs <= 1e-9)  # small tolerance for float noise
        return is_monotonic, diffs


# =====================================================================
# Entry point — wire everything together once your TODOs are filled in
# =====================================================================
if __name__ == "__main__":
    # Load samples the same way your offline's fs_redrawer.py does
    # (e.g. from svgs/heart.svg -> parametrized t, f_vals). Swap this stub
    # out for however your offline extracts samples.
    M = 500
    T = 1.0
    t = np.linspace(0, T, M, endpoint=False)
    # placeholder ground truth — replace with your actual SVG-derived signal
    f_vals = np.exp(1j * 2 * np.pi * t) + 0.3 * np.exp(1j * 2 * np.pi * 5 * t)

    N = 150
    epi = FourierEpicycles(t, f_vals, N, T)
    epi.coeffs = epi.compute_coeffs()

    pruner = TopKPruner()
    sweep = MSESweep()
    validator = PruningValidator()

    ok, mse0 = validator.check_full_reconstruction_zero_error(epi)
    print(f"[full reconstruction ~0 error] valid={ok} mse={mse0:.2e}")

    k = 20
    ok, count = validator.check_top_k_count_and_zeros(pruner, epi, k)
    print(f"[top-{k} pruning] valid={ok} nonzero_count={count}")

    n_values = [5, 10, 20, 50, 100, 150]
    mse_by_n = sweep.mse_vs_n_sweep(epi, n_values)
    for n in n_values:
        print(f"  N cutoff={n:>3}  MSE={mse_by_n[n]:.6e}")
    ok, diffs = validator.check_mse_monotonic_nonincreasing(mse_by_n)
    print(f"[MSE vs N monotonic non-increasing] valid={ok}")

    k_values = [5, 10, 20, 50, 100, 150]
    mse_by_k = sweep.mse_vs_k_sweep(epi, pruner, k_values)
    for k in k_values:
        print(f"  K={k:>3}  MSE={mse_by_k[k]:.6e}")
    ok, diffs = validator.check_mse_monotonic_nonincreasing(mse_by_k)
    print(f"[MSE vs K monotonic non-increasing] valid={ok}")
