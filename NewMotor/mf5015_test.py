import numpy as np
from NewMotor.mf5015 import CANdev
import multiprocessing
import time

if __name__ == '__main__':
    can_dict = multiprocessing.Manager().dict()
    can_dict['heave_1'] = 0.0
    can_dict['heave_2'] = 0.0
    can_dict['pitch_1'] = 0.0
    can_dict['pitch_2'] = 0.0
    #time.sleep(1.0)
    can = CANdev(can_dict)
    can.start()

    t0= time.time()
    while True:
        ## pitch motion, degree
        can_dict['pitch_1'] = -1 * 0 * 3.0*np.sin(0.1*(time.time()-t0))
        ## heave motion, mm
        can_dict['heave_1'] = 0 * 4.5*np.sin(0.1*(time.time()-t0))
        #can_dict['heave_2'] = 180*np.sin(2*np.pi/1.00*(now-time.perf_counter()))
        #can_dict['pitch_2'] = 30 * 3.0*np.sin(2*np.pi/1.00*(now-time.perf_counter()))
        #time.sleep(0.01)


