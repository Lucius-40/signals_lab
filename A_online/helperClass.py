import matplotlib.pyplot as plt
import numpy as np

from signal_lti import DiscreteSignal, LTISystem, readable_time_ticks

class Helper : 
    @staticmethod
    def make_signal(start_time, end_time, values):
        x = DiscreteSignal(start_time, end_time)
        for val, i in zip(values , range(start_time, end_time + 1)):
            x.set_value_at_time(i,val)

        return x

# Build a DiscreteSignal from selected sample values.
    @staticmethod
    def signal_from_samples(start_time, end_time, samples):
        times = [i for i, j in samples]
        maximum = max(end_time, max(times))
        minimum = min(start_time, min(times))
        x = DiscreteSignal(minimum, maximum)
        for i, j in samples:
            x.set_value_at_time(i, j)
        return x


# Return the identity impulse response: h[0] = 1.
    @staticmethod
    def impulse_identity():
        x = DiscreteSignal(0,0)
        x.set_value_at_time(0,1)
        return x


# Return first difference: h[0] = 1, h[1] = -1.
    @staticmethod
    def impulse_first_difference():
        x = DiscreteSignal(0, 1)
        x.set_value_at_time(0, 1)
        x.set_value_at_time(1, -1)
        return x

    @staticmethod
    def print_signal(signal, name):
        lines = [f"{name}:"]
        for n in signal.times():
            lines.append(f"  n = {n:4d}, value = {signal.get_value_at_time(n):10.4f}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def plot_signals_as_stems(input_signal, impulse_response, output_signal, save_path):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

        input_signal.plot("Input signal x[n]", ax=axes[0])
        impulse_response.plot("Impulse response h[n]", ax=axes[1])
        output_signal.plot("Output signal y[n]", ax=axes[2])

        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def max_absolute_difference(first_signal, second_signal):
        s = min(first_signal.start_time, second_signal.start_time)
        e = max(first_signal.end_time, second_signal.end_time)
        diff = 0
        for i in range(s, e + 1):
            fi = first_signal.get_value_at_time(i)
            si = second_signal.get_value_at_time(i)
            diff = max(diff, abs(fi - si))
        return diff


