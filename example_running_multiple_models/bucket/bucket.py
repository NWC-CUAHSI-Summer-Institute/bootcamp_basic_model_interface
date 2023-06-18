import time
import numpy as np
import pandas as pd

class BUCKET():
    def __init__(self):
        super(BUCKET, self).__init__()
        
    # __________________________________________________________________________________________________________
    # MAIN MODEL FUNCTION
    def run_bucket(self, b):
        """
            This is a simple model of a bucket with a hole (outlet) in it
            Args:
                self (class): This is the model class
                b (dictionary): This has all the information to run the model, including fluxes.
                    input_m
                    water_level_m
                    max_water_surface_elevation_m
                    potential_et_m_per_timestep
                    actual_et_m_per_timestep

                    total_in_m3
                    total_lost_m3
            Returns:
                Nothing, just updates the values in the b
        """
        
        # ________________________________________________
        b.input_m = (b.input_mm / 1000) * b.time_step_size
        if b.input_m > 0:
            b.total_in += b.input_m
       
       # Add the input mass to the bucket
        b.water_level_m = b.water_level_m + b.input_m

        # ________________________________________________
        # First check to see what goes over the bucket
        
        # Overflow if the bucket is too full
        if b.water_level_m > b.max_water_surface_elevation_m:
            # TODO: ADD WEIR EQUATION
            b.overflow_m = (b.water_level_m - b.max_water_surface_elevation_m) * 0.95
        else:
            b.overflow_m = 0
        # ________________________________________________
        b.water_level_m = b.water_level_m - b.overflow_m
        b.overflow_m3 = b.overflow_m * b.bucket_top_area_m2
        b.total_overflow += b.overflow_m
        
        
        # ________________________________________________
        b.potential_et_m_per_timestep = b.potential_et_m_per_s * b.time_step_size
        
        b.actual_et_m_per_timestep = np.max([0,np.min(np.array([b.potential_et_m_per_timestep, b.water_level_m]))])
        
        b.water_level_m = b.water_level_m - b.actual_et_m_per_timestep
        
        b.total_lost += b.actual_et_m_per_timestep
        


        # Calculate head on the outlet
        b.head_over_outlet_m = (b.water_level_m - b.outlet_m)

        # Calculate water leaving bucket through outlet
        if b.head_over_outlet_m > 0:
            
            b.velocity_out_m_per_s = np.sqrt(2 * b.g * b.head_over_outlet_m)
            
            b.outlet_m3 = b.discharge_coefficient * b.velocity_out_m_per_s *  b.outlet_cross_area_m2 * b.time_step_size
            b.outlet_m = b.outlet_m3 / b.bucket_top_area_m2
        
        else:

            b.outlet_m3= 0
            b.outlet_m = 0
        
        b.total_outlet += b.outlet_m
        
        # ________________________________________________
        b.water_level_m = b.water_level_m - b.outlet_m

        # ________________________________________________
        b.current_time_step += 1
        b.current_time      += b.time_step_size

        return
    

