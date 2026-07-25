import numpy as np



def readable_time_ticks(time_values, max_labels=18):
    if len(time_values) <= max_labels:
        return time_values

    step = int(np.ceil(len(time_values) / max_labels))
    ticks = time_values[::step]

    if ticks[-1] != time_values[-1]:
        ticks.append(time_values[-1])

    return ticks


class DiscreteSignal:
    """Finite discrete-time signal with integer indices."""

    # Create a finite discrete-time signal over the given integer range.
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.t = np.arange(start_time, end_time+1, 1)
        self.x = np.zeros(len(self.t))

    # Convenience alias used by plotting / visualization code.
    @property
    def values(self):
        return self.x

    # Return the number of stored samples in the signal.
    def __len__(self):
        return len(self.x)

    # Return the integer time indices covered by the signal.
    def times(self):
        return self.t

    # Return the signal value at the given time index.
    def get_value_at_time(self, t):
        if t > self.end_time or t < self.start_time :
            return 0
        index = t - self.start_time
        return self.x[index]

    # Set the signal value at the given time index.
    def set_value_at_time(self, t, value):
        if t > self.end_time or t < self.start_time :
            return 
        index = t - self.start_time
        self.x[index] = value

    # Return a shifted copy of the signal.
    def shift(self, k):
        shifted = DiscreteSignal(self.start_time + k, self.end_time + k)
        shifted.x = self.x.copy()
        return shifted

    # Return the sum of this signal and another signal.
    def add(self, other):
        start = min(self.start_time, other.start_time)
        end = max(self.end_time, other.end_time)

        y = DiscreteSignal(start, end)
        for i in range(start, end+1) :
            xi = self.get_value_at_time(i)
            xj = other.get_value_at_time(i)
            yi = xi + xj
            y.set_value_at_time(i,yi) 

        return y 

    # Return a scaled copy of the signal.
    def multiply(self, scalar):
        y = DiscreteSignal(self.start_time, self.end_time)
        y.x = scalar* self.x 
        return y 

    # Return the nonzero samples of the signal.
    def nonzero_samples(self, tolerance=1e-12):
        samples = []
        for n,val in zip(self.t, self.x) :
            if abs(val) > tolerance:
                samples.append((int(n), float(val)))
        return samples

    def plot(self, title, save_path=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        time_values = list(self.times())
        markerline, stemlines, baseline = ax.stem(time_values, self.values)
        markerline.set_markersize(6)
        baseline.set_color("black")
        baseline.set_linewidth(1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.35)
        ax.set_xticks(readable_time_ticks(time_values))
        ax.tick_params(axis="x", labelsize=9)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax


 
