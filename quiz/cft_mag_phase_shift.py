import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
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
            Compute the real and imaginary parts of the 2D Continuous Fourier
            Transform of self.I, using SEPARABLE trapezoidal integration:
    
                Re{F(u,v)} =  Integral Integral I(x,y) cos(2*pi*(u*x + v*y)) dx dy
                Im{F(u,v)} = -Integral Integral I(x,y) sin(2*pi*(u*x + v*y)) dx dy
    
            Do NOT evaluate this as a direct 4-nested-loop double integral over
            (x, y, u, v) -- that is O(N^4) and will not finish in reasonable
            time. Instead exploit separability: expand cos(2*pi*(ux+vy)) and
            sin(2*pi*(ux+vy)) with the angle-sum identities, first integrate
            over x for every (y, u) pair, then integrate the result over y for
            every (u, v) pair. Each of the two stages is an O(N^3) operation
            (an O(N) numerical integral, repeated over an N x N grid), which is
            what makes this tractable.
    
            Use self.u and self.v (NOT self.x/self.y) as the frequency axes --
            they were already computed for you in __init__.
    
            Use np.trapezoid(..., axis=...) for the integration -- no built-in
            FFT/DFT routine (np.fft, scipy.fft, ...) may be used anywhere in
            this method.
    
            Returns
            -------
            real, imag : two 2D numpy arrays, each of shape self.I.shape
            """
            # TODO: implement this method
            # Firstly, we need an (n,m) array for each u
            helper = Helper(self)
            g1_p = np.array([helper.compute_g1_p_foreach_u(u) for u in self.u])
            #now we need g for each v
            g1 = np.array([helper.compute_g1_foreach_v(v, g1_p) for v in self.v])
    
            g2_p = np.array([helper.compute_g2_p_foreach_u(u) for u in self.u])
            g2 = np.array([helper.compute_g2_foreach_v(v,g2_p) for v in self.v])
    
            real = g1 - g2 
    
            #Now the imaginary part : 
    
            f1_p = g2_p
            f1 = np.array([helper.compute_f1_foreach_v(v, f1_p) for v in self.v])
            
            f2_p = g1_p
            f2 = np.array([helper.compute_f2_foreach_v(v,f2_p) for v in self.v])
    
            im = f1 + f2 
            im = (-1) * im
    
            return real, im
            
    
    def plot_magnitude(self):
            """
            Plot the log-scaled magnitude spectrum of the 2D CFT computed by
            compute_cft(), i.e. plt.imshow(np.log(1 + magnitude), ...) where
            magnitude = sqrt(real**2 + imag**2). Purely for your own visual
            debugging -- not called by the command-line entry point below.
            """
            # TODO: implement this method
            real, imaginary = self.compute_cft()
            magnitude = np.sqrt(real**2 + imaginary**2)
            plt.imshow(np.log(1 + magnitude), cmap='gray')
            plt.title("Magnitude Spectrum")
            plt.axis('off')
            plt.show()

class Helper : 
    def __init__(self, cft2D : CFT2D):
        self.cft2D = cft2D

    #This is for the real block
    def compute_g1_p_foreach_u(self, u):
        c = np.cos(2 * np.pi* u*self.cft2D.x)
        integ = self.cft2D.I * c 
        return np.trapezoid(integ, x=self.cft2D.x, axis=1)
    
    def compute_g1_foreach_v(self, v, I):
        c = np.cos(2 * np.pi* v*self.cft2D.y)
        integ = I * c 
        return np.trapezoid(integ, x=self.cft2D.y, axis=1)
    
    def compute_g2_p_foreach_u(self, u):
        c = np.sin(2 * np.pi* u*self.cft2D.x)
        integ = self.cft2D.I * c 
        return np.trapezoid(integ, x=self.cft2D.x, axis=1)
        
    def compute_g2_foreach_v(self, v, I):
        c = np.sin(2 * np.pi* v*self.cft2D.y)
        integ = I * c 
        return np.trapezoid(integ, x=self.cft2D.y, axis=1)

    

        
    #this is for the imaginary block


    
    def compute_f1_foreach_v(self, v, I):
        c = np.cos(2 * np.pi* v*self.cft2D.y)
        integ = I * c 
        return np.trapezoid(integ, x=self.cft2D.y, axis=1)
            
    def compute_f2_foreach_v(self, v, I):
        c = np.sin(2 * np.pi* v*self.cft2D.y)
        integ = I * c 
        return np.trapezoid(integ, x=self.cft2D.y, axis=1)


class InverseCFT2D:
    """Reconstructs the spatial-domain image from a (filtered) 2D frequency
    spectrum using separable numerical integration."""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Perform the inverse 2D Continuous Fourier Transform:

            I(x,y) = Integral Integral F(u,v) exp(j*2*pi*(u*x + v*y)) du dv

        using the same separable-integration strategy as compute_cft():
        expand the complex exponential into cos/sin via Euler's identity,
        integrate over v first (for every (y, u) pair), then integrate
        that result over u (for every (x, y) pair). Use np.trapezoid.

        self.real, self.imag are the (possibly filtered) frequency-domain
        components; self.u, self.v are the frequency axes they were
        computed on; self.x, self.y are the spatial axes to reconstruct
        onto.

        Returns
        -------
        image : 2D numpy array of shape (len(self.y), len(self.x))
            The reconstructed real-valued spatial-domain signal. Note
            that after a high-pass filter this is NOT a valid image on
            its own (it will contain negative values, since the DC/
            low-frequency component that carried the average brightness
            has been removed) -- see the command-line entry point below
            for how it gets turned into a displayable edge map.
        """
        # TODO: implement this method
        helper = ReconstructionHelper(self)

        p1_p = np.array([helper.compute_p1_p_foreach_x(x) for x in self.x])
        p1 = np.array([helper.compute_p1_foreach_y(y, p1_p) for y in self.y])

        p2_p = np.array([helper.compute_p2_p_foreach_x(x) for x in self.x])
        p2 = np.array([helper.compute_p2_foreach_y(y, p2_p) for y in self.y])

        p = p1 - p2 

        q1_p = np.array([helper.compute_q1_p_foreach_x(x) for x in self.x])
        q1 = np.array([helper.compute_p1_foreach_y(y, q1_p) for y in self.y])

        q2_p = np.array([helper.compute_q2_p_foreach_x(x) for x in self.x])
        q2 = np.array([helper.compute_p2_foreach_y(y,q2_p) for y in self.y])

        q = q1 + q2

        return p-q 



class ReconstructionHelper : 
    def __init__(self, inv : InverseCFT2D):
        self.inv = inv

    def compute_p1_p_foreach_x(self, x):
        c = np.cos(2 * np.pi* x*self.inv.u)
        integ = self.inv.real * c
        return np.trapezoid(integ, x=self.inv.u, axis=1)
        
    def compute_p1_foreach_y(self, y, I):
        c = np.cos(2 * np.pi* y*self.inv.v)
        integ = I * c 
        return np.trapezoid(integ, x=self.inv.v, axis=1)
        
    def compute_p2_p_foreach_x(self, x):
        c = np.sin(2 * np.pi* x*self.inv.u)
        integ = self.inv.real * c 
        return np.trapezoid(integ, x=self.inv.u, axis=1)
            
    def compute_p2_foreach_y(self, y, I):
        c = np.sin(2 * np.pi* y*self.inv.v)
        integ = I * c 
        return np.trapezoid(integ, x=self.inv.v, axis=1)



    #For the im part 


    def compute_q1_p_foreach_x(self, x):
        c = np.sin(2 * np.pi* x*self.inv.u)
        integ = self.inv.imag * c
        return np.trapezoid(integ, x=self.inv.u, axis=1)
        
# def compute_q1_foreach_y(self, y, I):
#     c = np.cos(2 * np.pi* y*self.inv.v)
#     integ = I * c 
#     return np.trapezoid(integ, x=self.inv.v, axis=1)
        
    def compute_q2_p_foreach_x(self, x):
        c = np.cos(2 * np.pi* x*self.inv.u)
        integ = self.inv.imag * c 
        return np.trapezoid(integ, x=self.inv.u, axis=1)
            
# def compute_p2_foreach_y(self, y, I):
#     c = np.sin(2 * np.pi* y*self.inv.v)
#     integ = I * c 
#     return np.trapezoid(integ, x=self.inv.v, axis=1)

    




# =====================================================================
# Task A — Magnitude/Phase decomposition & swap
# =====================================================================
class MagnitudePhaseTools:
    """Practice: decompose a spectrum into magnitude/phase, recombine, and swap."""

    def to_mag_phase(self, real, imag):
        """TODO: return (magnitude, phase) arrays from real/imag spectrum components."""
        magnitude = np.abs(real + 1j*imag)
        phase = np.angle(real+ 1j*imag)
        return (magnitude, phase)


    def from_mag_phase(self, magnitude, phase):
        """TODO: return (real, imag) reconstructed from magnitude/phase arrays."""
        numbers = magnitude*np.exp(1j*phase)
        real = np.real(numbers)
        imag = np.imag(numbers)
        return (real, imag)

    def swap_magnitude_phase(self, real_a, imag_a, real_b, imag_b):
        """
        TODO: build a hybrid spectrum using the MAGNITUDE of spectrum A
        and the PHASE of spectrum B. Return (real, imag) of the hybrid.
        """
        
        a_mag, a_phase = self.to_mag_phase(real_a, imag_a)
        b_mag, b_phase = self.to_mag_phase(real_b, imag_b)
        real, imag = self.from_mag_phase(a_mag, b_phase)
        return real, imag 

    def magnitude_only(self, real, imag):
        """TODO: zero out phase (i.e. treat as purely real, non-negative), keep magnitude only."""
        mag = np.sqrt(real**2 + imag**2)
        phase = np.zeros(mag.shape)
        return self.from_mag_phase(mag, phase)

    def phase_only(self, real, imag):
        """TODO: set magnitude to 1 everywhere, keep phase only."""
        phase = np.angle(real + 1j*imag)
        mag = np.ones_like(phase)
        return self.from_mag_phase(mag,phase)


# =====================================================================
# Task B — Parseval's theorem check
# =====================================================================

class ParsevalChecker:
    """Practice: verify energy conservation between spatial and frequency domains."""

    def spatial_energy(self, image):
        """TODO: return sum(|I(x,y)|^2) over the spatial-domain image."""
        dx = img_a.x[1] - img_a.x[0]
        dy = img_a.y[1] - img_a.y[0]
        return np.sum(image**2)*dx*dy

    def frequency_energy(self, real, imag, du, dv):
        """
        TODO: return the frequency-domain energy sum(|F(u,v)|^2) * du * dv
        (match whatever normalization your offline's compute_cft convention uses).
        """
        mag = np.sum(real**2 + imag**2) 
        return mag * du*dv

    def verify_parseval(self, image, real, imag, du, dv, tol=1e-6):
        """
        TODO: compare spatial_energy(image) to frequency_energy(real, imag, du, dv).
        Return (is_valid: bool, relative_error: float).
        """
        spatial = self.spatial_energy(image)
        freq = self.frequency_energy(real, imag, du, dv)
        return (np.abs(spatial-freq) < tol, np.abs(spatial-freq))
# =====================================================================
# Task C — Shift theorem
# =====================================================================

class ShiftTheorem:
    """Practice: translating an image in space <=> multiplying its spectrum by a phase ramp."""

    def shift_image_spatial(self, image, x, y, dx, dy):
        """
        Shift image by (dx, dy) on the given x,y grid using bilinear interpolation.
        Positive dx moves content toward +x, positive dy toward +y.
        """
        H, W = image.shape
        x_src = x - dx
        y_src = y - dy

        shifted = np.zeros_like(image, dtype=float)

        for iy in range(H):
            ys = y_src[iy]
            if ys < y[0] or ys > y[-1]:
                continue

            y1 = np.searchsorted(y, ys)
            y0 = max(0, y1 - 1)
            y1 = min(H - 1, y1)

            if y1 == y0:
                wy = 0.0
            else:
                wy = (ys - y[y0]) / (y[y1] - y[y0])

            for ix in range(W):
                xs = x_src[ix]
                if xs < x[0] or xs > x[-1]:
                    continue

                x1 = np.searchsorted(x, xs)
                x0 = max(0, x1 - 1)
                x1 = min(W - 1, x1)

                if x1 == x0:
                    wx = 0.0
                else:
                    wx = (xs - x[x0]) / (x[x1] - x[x0])

                # bilinear interpolation
                v00 = image[y0, x0]
                v01 = image[y0, x1]
                v10 = image[y1, x0]
                v11 = image[y1, x1]

                shifted[iy, ix] = (
                    (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                    wy * ((1 - wx) * v10 + wx * v11)
                )

        return shifted

    def apply_phase_ramp(self, real, imag, u, v, dx, dy):
        """
        Multiply F(u,v) by exp(-i*2*pi*(u*dx + v*dy)).
        """
        F = real + 1j * imag
        U, V = np.meshgrid(u, v)  # shape: (len(v), len(u))
        ramp = np.exp(-1j * 2 * np.pi * (U * dx + V * dy))
        F_shifted = F * ramp
        return np.real(F_shifted), np.imag(F_shifted)

    def verify_shift_theorem(self, cft2d_obj: CFT2D, image_obj: ContinuousImage,
                              dx, dy, tol=1e-6):
        """
        Compare direct CFT of shifted image vs phase-ramp shifted spectrum.
        Return (is_valid, max_abs_delta).
        """
        # 1) original spectrum
        real, imag = cft2d_obj.compute_cft()

        # 2) direct CFT of spatially shifted image
        shifted_img = self.shift_image_spatial(image_obj.image, image_obj.x, image_obj.y, dx, dy)

        # lightweight image-like wrapper with same x/y grid
        class _TmpImage:
            pass

        tmp = _TmpImage()
        tmp.image = shifted_img
        tmp.x = image_obj.x
        tmp.y = image_obj.y

        cft_shifted = CFT2D(tmp)
        real_s, imag_s = cft_shifted.compute_cft()

        # 3) phase-ramp shift in frequency domain
        real_p, imag_p = self.apply_phase_ramp(real, imag, cft2d_obj.u, cft2d_obj.v, dx, dy)

        # 4) compare
        delta = np.max(np.abs((real_s + 1j * imag_s) - (real_p + 1j * imag_p)))
        return (delta <= tol), delta


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
