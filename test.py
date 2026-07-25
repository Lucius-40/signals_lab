from convolution.signal_lti import DiscreteSignal, LTISystem, readable_time_ticks
from convolution.helperClass import Helper
import numpy as np

x = np.array([1,2,3,4,5,6])
t = np.array([0,4,5,6,7,8])

y = np.array([11,12,3,4,15,6])

samples = list(zip(t,x))
signal = Helper.signal_from_samples(12,3,samples)
system = LTISystem(signal)
signal_in =Helper.signal_from_samples(12,3,list(zip(t,y)))

output= system.output(signal_in)

print(Helper.print_signal(output, "Test Signal"))