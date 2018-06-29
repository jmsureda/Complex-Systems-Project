
# coding: utf-8

# In[ ]:


import numpy as np

class Body:
    
    '''
    A class container for the bodies to be simulated. 
    Our bodies will be characterized by it's mass, position, velocity and acceleration.
    
    The arguments to initialize each body will be:
    
    - position
    - velocity
    - mass
    - name
    
    The acceleration of each body will be initialized as zero and, once all the bodies are created,
    then the acceleration it has to be computed for each body.
    
    '''
    
    def __init__(self,position,velocity,mass, name = ''):
        
        #Position n
        self.position = position
        
        #Velocity n
        self.velocity = velocity
    
        #Acceleration
        self.acceleration_i = np.zeros((3,))
        
        #Acceleration on the next time step
        self.acceleration_i_1 = np.zeros((3,))
        
        #Mass
        self.mass = mass
        
        #Name 
        self.name = name
        
        # N iterations
        
        self.iter = 0 #This will be useful when saving data, mainly when using a pickle.
    
    def Compute_acceleration(self,Bodies):
        
         
        G = 39.478 # Gravitational Constant in units AU^3 M_sol-1 yr^-2
        
        epsilon=0.01 #Softening Parameter
        
        N = len(Bodies)
        
        a = np.zeros((3,))
    
        for j in range(N):
        
            if Bodies[j] != self:
                
                direction = self.position - Bodies[j].position
                
                a_i = -(G*Bodies[j].mass*direction)/(np.linalg.norm(direction)**2+epsilon**2)**(3./2.)
                
                a += a_i    
            
        self.acceleration_i_1 = a 
        
        return self.acceleration_i_1
    
    def Compute_velocity(self):
        
        dt = 1./(36500) #yr   
        
        return self.velocity+0.5*(self.acceleration_i+self.acceleration_i_1)*dt     
    
    def Compute_position(self):
        
        dt = 1./(36500) #yr   
        
        self.iter += 1
        
        return self.position+self.velocity*dt+0.5*self.acceleration_i*dt**2

