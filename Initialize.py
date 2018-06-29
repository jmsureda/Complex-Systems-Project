import numpy as np
import pickle
from Body import Body

def Initialize(N, Pickle = 'None', L = 3, Center = 0, Sun = False):
    
    '''
    This function creates the bodies.
        
    In Args:
         - N:           Number of Bodies to be simulated.
         - Pickle:      (Optional) Recieves a pickle path to initialize the simulation from it.
         - L:           "Size" of the initial distribution on the position of the bodies. Default is 3 AU.
         - Center:       Center of the initial distribution of the bodies positions. Default is 0.
         - Sun:         (Optional) If 'True' adds a Sun-Like object at the center of the distribution. Default is False.
        
    Out Args:
        - Bodies:      Array containing the bodies to be simulated. 
        
    ** Note that if initialized using a pickle, the number of bodies initialized will be the same as the 
    bodies in the pickle **
        
    '''  
    if Pickle == 'None':
        
        
        Bodies = np.zeros((N,),dtype=object) #Define an array that will contain the Bodies.
        
        r = 2*L

        # Here we iterate to create each particle with a specific velocity and position.

        for i in range(N):

            # We choose this sigma considering that 3 sigma will represent the 99% of our distribution, 
            # then, when the majority of the particles wil be within the "scale" of the system.

            x = np.random.normal(Center,r/3.)
            y = np.random.normal(Center,r/3.)
            z = np.random.normal(Center,0.01) # This is to give the distribution a Disk-like form.

            pos = np.array([x,y,z])

            mass = np.random.uniform(0.001,1.)

            vx = np.random.uniform(-3,3)
            vy = np.random.uniform(-3,3)
            vz = np.random.uniform(-0.2,0.2)  #This is to give the distribution a Disk-like form.

            vel = np.array([vx,vy,vz])

            Bodies[i] = Body(pos, vel, mass, name='Particle_{}'.format(i)) #Create each body.
            
            if Sun == True: #Add a Main body in our system at the center. (Like a Sun)

                if i == N-1:

                    pos = np.array([Center,Center,Center])

                    mass = 5

                    vel = np.array([0,0,0])

                    Bodies[i] = Body(pos,vel,mass,name='Sun')

        #Here we compute the initial acceleration for the bodies.
        
        for i in range(N):

            Bodies[i].acceleration_i_1 = Bodies[i].Compute_acceleration(Bodies)
        
            Bodies[i].acceleration_i  = Bodies[i].acceleration_i_1 

        print('System initialized with %d bodies'%(N))

        return Bodies
        
        
    else: # If we initialize from a pickle file.
        
        Imported_Pickle = pickle.load(open(Pickle,'rb'))   
        
        Bodies = Imported_Pickle
        
        print('System initialized with %d bodies from a Pickle file.'%(len(Bodies)))
        
        return Bodies

