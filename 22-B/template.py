import numpy as np
import matplotlib.pyplot as plt

from signal_lti import DiscreteSignal, LTISystem, readable_time_ticks
from helperClass import Helper

# Todo: Define Signal class

# Todo: Define LTI class

if __name__ == "__main__":
    INF = 10

    x = DiscreteSignal(-5,5)
    x.set_value_at_time(0, 1)
    x.set_value_at_time(2, -1)
    x.plot("Input x(n)")

    h1 = DiscreteSignal(-1,1)
    h1.set_value_at_time(0, 1)

    h2 = DiscreteSignal(-1,2)
    h2.set_value_at_time(1, 0.5)

    h3 = DiscreteSignal(-2,2)
    h3.set_value_at_time(0, 1)
    h3.set_value_at_time(1, 1)

    sys1 = LTISystem(h1)
    sys2 = LTISystem(h2)
    sys3 = LTISystem(h3)
    
    # Todo: Determine output block by block
    y1= sys1.output(x)
    y2 = sys2.output(x)
    y_sum = y1.add(y2)
    y_final_1 = sys3.output(y_sum)

    y_final_1.plot("Output via block-by-block system")

    # Todo: Determine h_combined
    h = h1.add(h2)
    h = h.add(h3)
    h_combined = h 
    sys_combined = LTISystem(h_combined)

    y_final_2 = sys_combined.output(x)
    y_final_2.plot("Output via combined impulse response")

    print(Helper.print_signal(y_final_1, "By combination"))
    print("########### next : #############")

    print(Helper.print_signal(y_final_2, "Grouped together"))
