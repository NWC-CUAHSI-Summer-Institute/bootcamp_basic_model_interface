import time
import numpy as np
import pandas as pd

class PENMAN():
    def __init__(self):
        super(PENMAN, self).__init__()
        
    # __________________________________________________________________________________________________________
    # MAIN MODEL FUNCTION
    def run_penman(self, p):
        """
            This is a simple evapotranspiration model (Penman)
            Args:
                self (class): This is the model class
                e (dictionary): This has all the information to run the model.
            Returns:
                Nothing, just updates the values in the e
        """
        T_minus_T_d = 0.0023 * p.h + 0.37 * p.T * + 0.53 * p.R  + 0.35 * p.R_ann - 10.19
        penman_numerator = (700 * p.T_m) / (100 - p.A) + 15 * T_minus_T_d
        penman_denominator = 80 - p.T
        p.evaporation = penman_numerator - penman_denominator

        # ________________________________________________
        p.current_time_step += 1
        p.current_time      += p.time_step_size

        return
    

