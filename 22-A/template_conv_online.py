"""
CSE220 Online 2 (Step Response)

Instructions:
- Copy (or import) your completed Signal and LTI_System classes from Offline 1.
- Implement the TODO functions below.
- Do NOT use numpy.convolve / scipy.signal / any built-in convolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from signal_lti import DiscreteSignal, LTISystem, readable_time_ticks
from helperClass import Helper


# Paste/Import your Offline 1 implementations here
# from your_offline1_file import Signal, LTI_System

# class Signal:
#     def __init__(self, INF):
#         # TODO: paste your Offline 1 implementation
#         raise NotImplementedError

#     def set_value_at_time(self, t, value):
#         raise NotImplementedError

#     def shift(self, k):
#         raise NotImplementedError

#     def add(self, other):
#         raise NotImplementedError

#     def multiply(self, scalar):
#         raise NotImplementedError

#     def plot(self, title="Discrete Signal"):
#         raise NotImplementedError


# class LTI_System:
#     def __init__(self, impulse_response: Signal):
#         # TODO: paste your Offline 1 implementation
#         raise NotImplementedError

#     def linear_combination_of_impulses(self, input_signal: Signal):
#         raise NotImplementedError

#     def output(self, input_signal: Signal):
#         raise NotImplementedError


def read_signal_from_file(filename: str, INF: int) -> Signal:
    with open(filename, "r", encoding="utf-8") as f:
        nstart, nend = map(int, f.readline().strip().split())
        vals = list(map(float, f.readline().strip().split()))
    assert len(vals) == (nend - nstart + 1)
    sig = DiscreteSignal(nstart, nend)
    for i, v in enumerate(vals):
        sig.set_value_at_time(nstart + i, v)
    return sig


def first_difference(sig: DiscreteSignal) -> DiscreteSignal:
    """
    Returns Δsig[n] = sig[n] - sig[n-1] (assume outside range is 0).
    Must use Signal.shift/add/multiply.
    """
    signal1 = sig 
    signal2 = sig 
    signal2 = signal2.shift(1)
    signal2 = signal2.multiply(-1)
    return signal2.add(signal1)


def impulse_from_step_response(step_response: DiscreteSignal) -> DiscreteSignal:
    """
    Given s[n], compute h[n] = s[n] - s[n-1] (with s[-1]=0).
    Must use only Signal operations.
    """
    
    return first_difference(step_response)


def output_using_step_response(x: DiscreteSignal, step_response: DiscreteSignal) -> DiscreteSignal:
    """
    Compute y[n] using ONLY step response:
        y = (Δx * s)
    You must reuse your Offline 1 LTI_System machinery (linear combination of impulses).
    """
    # TODO
    del_x = first_difference(x)
    ltisystem = LTISystem(step_response)
    return ltisystem.output_by_superposition(del_x)


# Main (demo workflow)
if __name__ == "__main__":
    # Choose INF large enough for your signals
    INF = 50

    # ---- Load provided files ----
    s = read_signal_from_file("step_response.txt", INF)
    x = read_signal_from_file("input_signal.txt", INF)

    # ---- Part 1: recover impulse response ----
    h = first_difference(s)

    s.plot("Step Response s[n]")
    h.plot("Recovered Impulse Response h[n] = s[n] - s[n-1]")

    # ---- Part 2: output using only step response ----
    dx = first_difference(x)
    y_s = output_using_step_response(x,s)

    x.plot("Input x[n]")
    dx.plot("First Difference Δx[n]")
    y_s.plot("Output y_s[n] computed via step response")

    # ---- Part 3: verify with impulse-response method ----
    sys_h = LTISystem(h)
    y_h = sys_h.output(x)
    y_h.plot("Output y_h[n] computed via impulse response")

    # Check if outputs match closely
    if np.allclose(y_s.values, y_h.values, atol=1e-6):
        print("Outputs match closely!")
    else:
        print("Outputs differ!")
