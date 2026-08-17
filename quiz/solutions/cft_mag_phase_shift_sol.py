import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from scipy.ndimage import shift as ndi_shift
import sys


# =====================================================================
# Given classes — paste your offline implementations where indicated
# =====================================================================

class ContinuousImage:
    """Represents a grayscale image as a continuous 2D spatial signal. (Given)"""

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L').astype(float)
        self.image = self.image / np.max(self.image)
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])


class CFT2D:
    """2D Continuous Fourier Transform. (Given — paste your offline solution)"""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        """
        Approximate the continuous 2D FT via a Riemann sum:
            F(u,v) = sum_x sum_y I(x,y) * exp(-i*2*pi*(u*x + v*y)) * dx * dy
        Implemented as two matrix multiplications (separable exponential
        kernels) rather than np.fft so that F is evaluated exactly on the
        (possibly non-uniform-in-index) u, v grids defined in __init__.
        """
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        # Ex[k, m] = exp(-i*2*pi*u_k*x_m)   shape (Nu, Nx)
        Ex = np.exp(-1j * 2 * np.pi * np.outer(self.u, self.x))
        # Ey[k, m] = exp(-i*2*pi*v_k*y_m)   shape (Nv, Ny)
        Ey = np.exp(-1j * 2 * np.pi * np.outer(self.v, self.y))

        # F(v,u) = Ey @ I @ Ex.T  -> shape (Nv, Nu)
        F = Ey @ self.I @ Ex.T
        F = F * dx * dy

        self.real = F.real
        self.imag = F.imag
        return self.real, self.imag

    def plot_magnitude(self):
        if not hasattr(self, "real"):
            self.compute_cft()
        magnitude = np.abs(self.real + 1j * self.imag)
        log_mag = np.log1p(magnitude)

        plt.figure(figsize=(6, 6))
        plt.imshow(
            log_mag,
            extent=[self.u[0], self.u[-1], self.v[0], self.v[-1]],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        plt.colorbar(label="log(1 + |F(u,v)|)")
        plt.xlabel("u")
        plt.ylabel("v")
        plt.title("2D-CFT Magnitude Spectrum")
        plt.tight_layout()
        plt.savefig("cft_magnitude.png")
        plt.close()


class InverseCFT2D:
    """Inverse 2D-CFT. (Given — paste your offline solution)"""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Approximate the continuous inverse 2D FT via a Riemann sum:
            I(x,y) = sum_u sum_v F(u,v) * exp(+i*2*pi*(u*x + v*y)) * du * dv
        Implemented as two matrix multiplications, mirroring CFT2D.compute_cft.
        """
        F = self.real + 1j * self.imag
        du = self.u[1] - self.u[0]
        dv = self.v[1] - self.v[0]

        # Ex[m, k] = exp(+i*2*pi*x_m*u_k)   shape (Nx, Nu)
        Ex = np.exp(1j * 2 * np.pi * np.outer(self.x, self.u))
        # Ey[m, k] = exp(+i*2*pi*y_m*v_k)   shape (Ny, Nv)
        Ey = np.exp(1j * 2 * np.pi * np.outer(self.y, self.v))

        # I(y,x) = Ey @ F @ Ex.T -> shape (Ny, Nx)
        I_rec = Ey @ F @ Ex.T
        I_rec = I_rec * du * dv
        return I_rec


# =====================================================================
# Task A — Magnitude/Phase decomposition & swap
# =====================================================================

class MagnitudePhaseTools:
    """Practice: decompose a spectrum into magnitude/phase, recombine, and swap."""

    def to_mag_phase(self, real, imag):
        spectrum = real + 1j * imag
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        return magnitude, phase

    def from_mag_phase(self, magnitude, phase):
        spectrum = magnitude * np.exp(1j * phase)
        return spectrum.real, spectrum.imag

    def swap_magnitude_phase(self, real_a, imag_a, real_b, imag_b):
        magnitude_a, _ = self.to_mag_phase(real_a, imag_a)
        _, phase_b = self.to_mag_phase(real_b, imag_b)
        return self.from_mag_phase(magnitude_a, phase_b)

    def magnitude_only(self, real, imag):
        magnitude, _ = self.to_mag_phase(real, imag)
        # phase = 0 everywhere -> purely real, non-negative spectrum
        return self.from_mag_phase(magnitude, np.zeros_like(magnitude))

    def phase_only(self, real, imag):
        _, phase = self.to_mag_phase(real, imag)
        # magnitude = 1 everywhere
        return self.from_mag_phase(np.ones_like(phase), phase)


# =====================================================================
# Task B — Parseval's theorem check
# =====================================================================

class ParsevalChecker:
    """Practice: verify energy conservation between spatial and frequency domains."""

    def spatial_energy(self, image, dx=1.0, dy=1.0):
        """
        sum(|I(x,y)|^2) * dx * dy — a Riemann-sum approximation of
        integral |I(x,y)|^2 dx dy. dx, dy default to 1 so the raw,
        un-weighted pixel energy is still available if needed, but
        verify_parseval always passes the true grid spacing in.
        """
        return np.sum(np.abs(image) ** 2) * dx * dy

    def frequency_energy(self, real, imag, du, dv):
        """sum(|F(u,v)|^2) * du * dv, matching CFT2D.compute_cft's dx*dy-scaled convention."""
        spectrum = real + 1j * imag
        return np.sum(np.abs(spectrum) ** 2) * du * dv

    def verify_parseval(self, image, real, imag, du, dv, tol=1e-6):
        """
        Parseval: integral |I(x,y)|^2 dx dy == integral |F(u,v)|^2 du dv.
        Both sides here are Riemann-sum approximations, so both must carry
        their respective area elements (dx*dy on the spatial side, du*dv on
        the frequency side) — dx, dy are inferred from the image's own
        pixel count assuming the same [-1, 1] domain used by ContinuousImage.
        """
        ny, nx = image.shape
        dx = 2.0 / (nx - 1)
        dy = 2.0 / (ny - 1)
        e_spatial = self.spatial_energy(image, dx, dy)
        e_freq = self.frequency_energy(real, imag, du, dv)
        relative_error = np.abs(e_spatial - e_freq) / e_spatial
        is_valid = relative_error < tol
        return is_valid, relative_error


# =====================================================================
# Task C — Shift theorem
# =====================================================================

class ShiftTheorem:
    """Practice: translating an image in space <=> multiplying its spectrum by a phase ramp."""

    def shift_image_spatial(self, image, x, y, dx, dy):
        """
        Shift `image` by (dx, dy) in the continuous (x, y) coordinate system.
        Converts the continuous offsets into fractional pixel offsets using
        the grid spacing, then uses spline interpolation (order=1, i.e.
        bilinear) with wraparound, matching the implicit periodicity assumed
        by the DFT/CFT.
        """
        pixel_dx = dx / (x[1] - x[0])
        pixel_dy = dy / (y[1] - y[0])
        # image is indexed [row=y, col=x] -> shift is (row_shift, col_shift)
        # cubic spline (order=3) gives a much more accurate sub-pixel shift
        # than nearest/linear, which matters since the shift theorem check
        # compares against an exact phase-ramp prediction.
        shifted = ndi_shift(image, shift=(pixel_dy, pixel_dx), mode="wrap", order=3)
        return shifted

    def apply_phase_ramp(self, real, imag, u, v, dx, dy):
        U, V = np.meshgrid(u, v)  # U varies along columns, V along rows
        ramp = np.exp(-1j * 2 * np.pi * (U * dx + V * dy))
        spectrum = (real + 1j * imag) * ramp
        return spectrum.real, spectrum.imag

    def verify_shift_theorem(self, cft2d_obj: CFT2D, image_obj: ContinuousImage,
                              dx, dy, tol=1e-6):
        # 1. CFT of the original image
        real, imag = cft2d_obj.compute_cft()

        # 2. CFT of the spatially-shifted image, computed directly
        shifted_image = self.shift_image_spatial(
            image_obj.image, image_obj.x, image_obj.y, dx, dy
        )
        dx_grid = image_obj.x[1] - image_obj.x[0]
        dy_grid = image_obj.y[1] - image_obj.y[0]
        Ex = np.exp(-1j * 2 * np.pi * np.outer(cft2d_obj.u, image_obj.x))
        Ey = np.exp(-1j * 2 * np.pi * np.outer(cft2d_obj.v, image_obj.y))
        F_shifted = (Ey @ shifted_image @ Ex.T) * dx_grid * dy_grid
        real_s, imag_s = F_shifted.real, F_shifted.imag

        # 3. Predicted spectrum via the shift theorem (phase ramp)
        real_p, imag_p = self.apply_phase_ramp(real, imag, cft2d_obj.u, cft2d_obj.v, dx, dy)

        # 4. Compare
        delta = np.max(np.abs((real_s + 1j * imag_s) - (real_p + 1j * imag_p)))
        is_valid = delta < tol
        return is_valid, delta


# =====================================================================
# Validator — checks your filled-in methods against independent ground truth
# =====================================================================

class FrequencyDomainValidator:
    """
    Runs independent checks against your implementations above.
    Does NOT reimplement your algorithms — it checks the mathematical
    invariants your methods are supposed to satisfy.
    """

    def check_mag_phase_roundtrip(self, tools: MagnitudePhaseTools, real, imag, tol=1e-9):
        mag, phase = tools.to_mag_phase(real, imag)
        r2, i2 = tools.from_mag_phase(mag, phase)
        delta = np.max(np.abs((real + 1j * imag) - (r2 + 1j * i2)))
        return delta < tol, delta

    def check_magnitude_matches_original(self, tools: MagnitudePhaseTools, real, imag, tol=1e-9):
        """Swapping A's magnitude with A's own phase should reproduce A exactly."""
        r_swap, i_swap = tools.swap_magnitude_phase(real, imag, real, imag)
        delta = np.max(np.abs((real + 1j * imag) - (r_swap + 1j * i_swap)))
        return delta < tol, delta

    def check_phase_only_unit_magnitude(self, tools: MagnitudePhaseTools, real, imag, tol=1e-9):
        r_p, i_p = tools.phase_only(real, imag)
        mags = np.abs(r_p + 1j * i_p)
        delta = np.max(np.abs(mags - 1))
        return delta < tol, delta

    def check_parseval(self, checker: ParsevalChecker, image, real, imag, du, dv, tol=1e-3):
        is_valid, rel_err = checker.verify_parseval(image, real, imag, du, dv, tol=tol)
        return is_valid, rel_err

    def check_shift_theorem(self, shifter: ShiftTheorem, cft2d_obj, image_obj, dx, dy, tol=1e-3):
        is_valid, delta = shifter.verify_shift_theorem(cft2d_obj, image_obj, dx, dy, tol=tol)
        return is_valid, delta


# =====================================================================
# Entry point — wire everything together once your TODOs are filled in
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 cft_mag_phase_shift.py <image_a> <image_b>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    img_a = ContinuousImage(path_a)
    img_b = ContinuousImage(path_b)
    cft_a = CFT2D(img_a)
    cft_b = CFT2D(img_b)
    real_a, imag_a = cft_a.compute_cft()
    real_b, imag_b = cft_b.compute_cft()

    tools = MagnitudePhaseTools()
    validator = FrequencyDomainValidator()

    ok, d = validator.check_mag_phase_roundtrip(tools, real_a, imag_a)
    print(f"[mag/phase roundtrip] valid={ok} delta={d:.2e}")

    ok, d = validator.check_magnitude_matches_original(tools, real_a, imag_a)
    print(f"[swap A-mag with A-phase == A] valid={ok} delta={d:.2e}")

    ok, d = validator.check_phase_only_unit_magnitude(tools, real_a, imag_a)
    print(f"[phase-only has unit magnitude] valid={ok} delta={d:.2e}")

    real_hybrid, imag_hybrid = tools.swap_magnitude_phase(real_a, imag_a, real_b, imag_b)
    hybrid_img = InverseCFT2D(real_hybrid, imag_hybrid, cft_a.u, cft_a.v, img_a.x, img_a.y).reconstruct()
    plt.imsave("hybrid_magA_phaseB.png", np.clip(np.abs(hybrid_img), 0, 1), cmap='gray')
    print("Saved hybrid_magA_phaseB.png")

    checker = ParsevalChecker()
    du = cft_a.u[1] - cft_a.u[0]
    dv = cft_a.v[1] - cft_a.v[0]
    ok, rel_err = validator.check_parseval(checker, img_a.image, real_a, imag_a, du, dv)
    print(f"[Parseval] valid={ok} relative_error={rel_err:.2e}")

    shifter = ShiftTheorem()
    ok, d = validator.check_shift_theorem(shifter, cft_a, img_a, dx=0.1, dy=0.05)
    print(f"[Shift theorem] valid={ok} max_abs_delta={d:.2e}")
