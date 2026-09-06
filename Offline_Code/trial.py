import numpy as np
from transforms import DFTAnalyzer, FFTTransformer, next_power_of_two

d = DFTAnalyzer()
f = FFTTransformer()

# 1. DFT and FFT must agree on the same input (real signal)
x = np.random.randn(64)
assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9, "DFT/FFT mismatch on real input"
print("Test 1 passed: DFT and FFT agree on real input")

# 2. DFT and FFT must agree on complex input too
x = np.random.randn(32) + 1j * np.random.randn(32)
assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9, "DFT/FFT mismatch on complex input"
print("Test 2 passed: DFT and FFT agree on complex input")

# 3. Round-trip: inverse(transform(x)) should recover x, for both engines
x = np.random.randn(16) + 1j * np.random.randn(16)
assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9, "DFT round-trip failed"
assert np.max(np.abs(f.inverse(f.transform(x)) - x)) < 1e-9, "FFT round-trip failed"
print("Test 3 passed: round-trip recovers original signal (both engines)")

# 4. Impulse input: DFT of a delta function should be all ones (known analytic result)
N = 8
impulse = np.zeros(N, dtype=complex)
impulse[0] = 1
expected = np.ones(N, dtype=complex)
assert np.max(np.abs(f.transform(impulse) - expected)) < 1e-9, "Impulse response incorrect"
print("Test 4 passed: impulse input produces all-ones spectrum")

# 5. N=1 edge case: transform of a single value should return that value unchanged
x = np.array([3.0 + 4.0j])
assert np.max(np.abs(f.transform(x) - x)) < 1e-9, "N=1 case failed"
assert np.max(np.abs(f.inverse(x) - x)) < 1e-9, "N=1 inverse failed"
print("Test 5 passed: N=1 trivial case handled correctly")

# 6. Non-power-of-two length must raise ValueError, not silently produce wrong output
try:
    f.transform(np.random.randn(100))
    raise AssertionError("Expected ValueError for N=100, but no exception was raised")
except ValueError:
    print("Test 6 passed: non-power-of-two length correctly raises ValueError")

# 7. All-zero input: should produce an all-zero spectrum (not NaN/inf from any stray division)
x = np.zeros(16, dtype=complex)
assert np.max(np.abs(f.transform(x))) < 1e-9, "Zero input did not produce zero spectrum"
print("Test 7 passed: all-zero input handled cleanly")
import numpy as np
from transforms import DFTAnalyzer, FFTTransformer

d = DFTAnalyzer()
f = FFTTransformer()

# 8. Constant signal -> DFT is a spike at bin 0 equal to N*constant, all else 0
# Reasoning: sum of N identical values, k=0 term has no oscillation (exp(0)=1)
N = 8
x = np.full(N, 3.0, dtype=complex)
expected = np.zeros(N, dtype=complex)
expected[0] = 3.0 * N  # = 24
assert np.max(np.abs(f.transform(x) - expected)) < 1e-9, "Constant signal spectrum wrong"
print("Test 8 passed: constant signal produces spike at bin 0")

# 9. Pure cosine at bin 1 -> energy should appear ONLY at bins 1 and N-1
# x[n] = cos(2*pi*n/N) = 0.5*exp(j2pi n/N) + 0.5*exp(-j2pi n/N)
# So X[1] = N/2, X[N-1] = N/2, everything else ~0
N = 16
n = np.arange(N)
x = np.cos(2 * np.pi * n / N)
X = f.transform(x)
expected = np.zeros(N, dtype=complex)
expected[1] = N / 2
expected[N - 1] = N / 2
assert np.max(np.abs(X - expected)) < 1e-9, "Single-frequency cosine spectrum wrong"
print("Test 9 passed: cosine at bin 1 produces spikes only at bins 1 and N-1")

# 10. Hand-computed N=4 DFT with a known worked-out answer
# x = [1, 2, 3, 4]
# X[0] = 1+2+3+4 = 10
# X[1] = 1 + 2*(-j) + 3*(-1) + 4*(j) = -2 + 2j
# X[2] = 1 - 2 + 3 - 4 = -2
# X[3] = 1 + 2*(j) + 3*(-1) + 4*(-j) = -2 - 2j
x = np.array([1, 2, 3, 4], dtype=complex)
expected = np.array([10, -2 + 2j, -2, -2 - 2j], dtype=complex)
assert np.max(np.abs(d.transform(x) - expected)) < 1e-9, "DFT hand-computed check failed"
assert np.max(np.abs(f.transform(x) - expected)) < 1e-9, "FFT hand-computed check failed"
print("Test 10 passed: N=4 output matches hand-computed values")

# 11. Parseval's theorem: energy in time domain == energy in frequency domain / N
# sum(|x[n]|^2) == (1/N) * sum(|X[k]|^2)
x = np.random.randn(32) + 1j * np.random.randn(32)
X = f.transform(x)
time_energy = np.sum(np.abs(x) ** 2)
freq_energy = np.sum(np.abs(X) ** 2) / len(x)
assert abs(time_energy - freq_energy) < 1e-9, "Parseval's theorem violated"
print("Test 11 passed: Parseval's theorem holds")

from transforms import DFTAnalyzer, FFTTransformer, ArbitraryLengthFFT

d = DFTAnalyzer()
f = FFTTransformer()
af = ArbitraryLengthFFT()

# 12. Core promise of the bonus: works on lengths that are NOT powers of two,
# and agrees with the brute-force DFT (the only ground truth available for
# non-power-of-two N, since FFTTransformer can't handle these lengths at all)
for N in [3, 5, 6, 7, 10, 15]:
    x = np.random.randn(N) + 1j * np.random.randn(N)
    assert np.max(np.abs(d.transform(x) - af.transform(x))) < 1e-9, \
        f"ArbitraryLengthFFT disagrees with DFT at N={N}"
print("Test 12 passed: ArbitraryLengthFFT matches DFT for several non-power-of-two N")

# 13. Round-trip for non-power-of-two N (catches the reversal/shape bugs
# directly, since a wrong b[] would still often produce SOME output but
# fail to invert correctly)
for N in [3, 5, 7, 11]:
    x = np.random.randn(N) + 1j * np.random.randn(N)
    recovered = af.inverse(af.transform(x))
    assert np.max(np.abs(recovered - x)) < 1e-9, f"Round-trip failed at N={N}"
print("Test 13 passed: round-trip recovers original signal for odd/prime N")

# 14. Must still work correctly when N happens to BE a power of two
# (the M = next_power_of_two(2N-1) padding logic shouldn't break this case)
x = np.random.randn(16) + 1j * np.random.randn(16)
assert np.max(np.abs(af.transform(x) - f.transform(x))) < 1e-9, \
    "ArbitraryLengthFFT disagrees with FFTTransformer when N is already a power of two"
print("Test 14 passed: correct even when N is already a power of two")

# 15. N=1 edge case (guards the early return path)
x = np.array([2.5 - 1.5j])
assert np.max(np.abs(af.transform(x) - x)) < 1e-9, "N=1 transform failed"
assert np.max(np.abs(af.inverse(x) - x)) < 1e-9, "N=1 inverse failed"
print("Test 15 passed: N=1 handled correctly")

# 16. Known hand-computable answer at a genuinely awkward N=3 length
# x = [1, 1, 0] -> X[k] = 1 + exp(-j*2*pi*k/3)
N = 3
x = np.array([1, 1, 0], dtype=complex)
k = np.arange(N)
expected = 1 + np.exp(-2j * np.pi * k / N)
assert np.max(np.abs(af.transform(x) - expected)) < 1e-9, \
    "N=3 hand-computed check failed"
print("Test 16 passed: N=3 matches hand-derived formula")

# 17. Regression guard for the specific bug you just fixed: an off-by-one or
# un-reversed negative-index block in b[] tends to corrupt just the HIGHER
# bins while bin 0 (the DC/sum term) still comes out right by coincidence.
# This test deliberately checks a non-DC bin against the brute-force DFT.
N = 6
x = np.arange(N, dtype=complex)
X_dft = d.transform(x)
X_arb = af.transform(x)
assert np.abs(X_arb[N - 1] - X_dft[N - 1]) < 1e-9, \
    "High-frequency bin wrong -- check b[] index ordering / reversal"
print("Test 17 passed: highest-frequency bin correct (guards b[] ordering bugs)")

print("\nAll ArbitraryLengthFFT tests passed.")