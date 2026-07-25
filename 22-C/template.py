import numpy as np
import matplotlib.pyplot as plt

from signal_lti import DiscreteSignal as Signal, readable_time_ticks
from helperClass import Helper
# Todo: Define Signal class
class LTISystem:
    """Discrete-time LTI system described by a finite impulse response."""

    # Store the impulse response that defines the LTI system.
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response

    # Return the output time range for the convolution result.
    def output_range(self, input_signal):
        s = input_signal.start_time + self.impulse_response.start_time
        e = input_signal.end_time + self.impulse_response.end_time
        return (s,e)

    # Return all shifted and scaled impulse-response components for the input.
    def get_response_components(self, input_signal):
        components = []
        for k , x_k in input_signal.nonzero_samples() :
            component = self.impulse_response.shift(k).multiply(x_k)
            components.append(component)
        return components
    # Return the system output using superposition of response components.
    def output_by_superposition(self, input_signal):
        components = self.get_response_components(input_signal)
        s,e = self.output_range(input_signal)
        y = Signal(s,e)

        for component in components :
            y = y.add(component)
        return y

    

    # Return the nonzero product terms that contribute to one output sample.
    def get_contributions_at_time(self, input_signal, n):
        """"
        For there to be contribution :
        1. k has to be in range of input , k between input s, input e
        2. n-k must be in range of impulse, k between h.s, h,e
        """
        h = self.impulse_response
        k_upper = max(input_signal.start_time, n-h.end_time)
        k_lower = min(input_signal.end_time, n - h.start_time)

        contributions =[]
        for k in range(k_upper, k_lower + 1):
            x_k = input_signal.get_value_at_time(k)
            h_n_k = h.get_value_at_time(n-k)
            val = x_k * h_n_k
            if val != 0 :
                contributions.append((k,val))
        return contributions

    # Return one output sample of the LTI system.
    def output_at_time(self, input_signal, n):
        sum = 0 
        for _, i in self.get_contributions_at_time(input_signal, n):
            sum = sum + i
        return sum

    # Return the complete output signal of the LTI system.
    def output(self, input_signal):
        s,e = self.output_range(input_signal)
        y = Signal(s,e)
        for n in range (s, e+ 1) : 
            y.set_value_at_time(n, self.output_at_time(input_signal, n))
        return y
    
    def output_super(self, superSignal : SuperSignal):
        y = Signal(-1,1)
        for coeff, signal in superSignal.components : 
            signal =signal.multiply(coeff)
            y = y.add(signal)
        return y
        
class SuperSignal:
    def __init__(self):
        self.components = []

    def add(self, signal: Signal, coefficient=1.0):
        self.components.append((coefficient, signal))
        
# Todo: Define LTI class

if __name__ == "__main__":
    INF = 10

    # Component signals
    x1 = Signal(-1,1)
    x1.set_value_at_time(0, 1)

    x2 = Signal(-1,5)
    x2.set_value_at_time(2, 1)

    
    # Todo: Create SuperSignal: x(n) = 2*x1(n) - x2(n)
    superSignal = SuperSignal()
    superSignal.add(x1,2)
    superSignal.add(x2,-1)

    # Impulse response
    h = Signal(-1,1)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    system = LTISystem(h)

    # Todo: Output using superposition

    output =  system.output_super(superSignal)
    print(Helper.print_signal(output, "Final Output"))
