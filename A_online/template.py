"""
Instructions:
- x1, x2, a, b, k, and h are already given below.
- Complete the TODOs.
- Do NOT use numpy.convolve / scipy.signal / any built-in convolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from signal_lti import DiscreteSignal,LTISystem,readable_time_ticks
from helperClass import Helper


def make_signal(start_time, end_time, values):
    """Helper: build a DiscreteSignal from a list of values."""
    signal = DiscreteSignal(start_time, end_time)
    for offset, value in enumerate(values):
        signal.set_value_at_time(start_time + offset, value)
    return signal


def max_absolute_difference(first_signal, second_signal):
    """Helper: largest |difference| between two signals over their combined range."""
    # TODO: reuse your offline implementation of this function.
    return Helper.max_absolute_difference(first_signal, second_signal)
    


# ---- Generic property testers ----
# These must work for ANY apply_system callable
# method such as system_a.output. Do not assume apply_system is an LTISystem.

# You shoulb be able to use this function like: test_linearity(sys_a.output, x1, x2, a, b) 
# or test_linearity(system_b, x1, x2, a, b)

def test_linearity(apply_system, x1, x2, a, b):

    #TODO: Return max| apply_system(a*x1 + b*x2)  -  (a*apply_system(x1) + b*apply_system(x2)) |
    a_x1 = x1.multiply(a)
    b_x2 = x2.multiply(b)
    sum = a_x1.add(b_x2)
    signal_1 = apply_system(sum)

    a_ax1 = apply_system(x1)
    a_ax1 = a_ax1.multiply(a)

    
    b_bx2 = apply_system(x2)
    b_bx2 = b_bx2.multiply(b)

    signal_2 = a_ax1.add(b_bx2)

    return max_absolute_difference(signal_1, signal_2)


    


def test_time_invariance(apply_system, x, k):

    #TODO: Return max| apply_system(x shifted by k)  -  (apply_system(x) shifted by k) |

    signal_1 = x.shift(k)
    signal_1 = apply_system(signal_1)

    signal_2 = apply_system(x)
    signal_2 = signal_2.shift(k)

    return max_absolute_difference(signal_2, signal_1)


# ---- System B: y[n] = n * x[n] ----

def system_b(input_signal):
    # TODO: build and return a DiscreteSignal where output[n] = n * input_signal[n]
    x = input_signal.values
    t = input_signal.times()
    n = len(x)
    for i in range (0, n) : 
        x[i] = i * x[i]
    y = Helper.signal_from_samples(t[0],t[n-1],list(zip(t,x)))
    return y



def main():
    tolerance = 1e-9

    # ---- Given signals and scalars (do not change) ----
    x1 = make_signal(-2, 2, [1, 0, 2, -1, 3])
    x2 = make_signal(-1, 3, [2, -3, 0, 1, 1])
    a, b = 2.0, -3.0
    k = 3

    h = make_signal(0, 2, [1.0, 0.5, 0.25])
    #TODO: Test both properties for system A

    print("=== System A: genuine LTI system (LTISystem.output) ===")
    impulse = DiscreteSignal(-1,1)
    impulse.set_value_at_time(0,1)
    lti_system = LTISystem(impulse)
    diff_linear_a = test_linearity(lti_system.output, x1,x2,a,b)
    diff_ti_a = test_time_invariance(lti_system.output, x1,k)
    print(f"Linearity max diff:        {diff_linear_a}")
    print(f"Time-invariance max diff:  {diff_ti_a}")

    print()

    #TODO: Test both properties for system B
    print("=== System B: y[n] = n * x[n] ===")
    diff_linear_b = test_linearity(system_b, x1,x2,a,b)
    diff_ti_b = test_time_invariance(system_b,x1,k)
    print(f"Linearity max diff:        {diff_linear_b}")
    print(f"Time-invariance max diff:  {diff_ti_b}")

    print()
    # TODO: print a short conclusion stating which property System B fails
    # (linearity or time-invariance).
    print("The statement is not linear")


if __name__ == "__main__":
    main()
