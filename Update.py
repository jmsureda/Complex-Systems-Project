import numpy as np
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import Visualization

def Compute_Accel(p_i,Bodies): # This function computes the acceleration (i+1)
    
    p_i.acceleration_i_1  = p_i.Compute_acceleration(Bodies)                 
            

def Update(Bodies, Steps = 100, Snaps = 10,L = 3, njobs= 1, Plots = False, direct = ''):
    
    '''
    This function updates the system.
        
    In Args:
         - Bodies:      Array containing the bodies to be updated.
         - *Steps:      (Optional) The number of update steps. If not given, it will update 100 steps.
         - *Snaps:      (Optional) The number of snapshots to take. This will only be useful when Plots = True.
         - *L = 3       The scale of the system. Default is 3 AU.
         - *Plots       (Optional) If true, will save certain stapshots on the given directory.
         - *direct      (Needed if Plots = True) A string with the Path where you want to save the outputs.
         - *njobs:      (Optional) The number of cores to be used during the computation. By default is 1
                        i.e. the program will run sequentially.
        
    Out Args:
        - Bodies:      Array containing the updated bodies. 
        
    '''      
    j=0 # This is a counter for the number of snapshots.
     
    for k in tqdm(range(Steps)):
        
        time.sleep(0.01)
        
        if Plots == True:
        
            N_steps_snapshot=int(Steps/Snaps)
        
            if k % N_steps_snapshot == 0:  # This will plot only the necesary steps.

                Visualization.Plot(Bodies,k,j,L,direct) # This will make a plot of the actual step of the simulation.

                j+=1
                
        B = Bodies # This is a "temporal array" that will serve to compute the acceleration in parallel
                   # without updating the positions of the bodies before we compute all the accelerations.
        
        Parallel(n_jobs=njobs)(delayed(Compute_Accel)(i,B) for i in Bodies)
        
        for i in range(len(Bodies)):
            
            Bodies[i].velocity = Bodies[i].Compute_velocity() 
    
            Bodies[i].position = Bodies[i].Compute_position()

            # Once the main parameters have been updated we define the acceleration (i) equal to the 
            # computed acceleration (i+1).

            Bodies[i].acceleration_i  = Bodies[i].acceleration_i_1    
    
    return Bodies

