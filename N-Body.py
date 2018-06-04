
# coding: utf-8

# # N-body Code using a brute force approach

# First, we import all the packages and modules that we'll be using 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
import time
import datetime
print('Modules Imported')


# Now, we define our class for the bodies to be simulated

# In[2]:


class Body:
    
    '''
    A class container for the bodies to be simulated. 
    Our bodies will be characterized by it's mass, position, velocity and acceleration.
    
    The arguments to initialize each body will be:
    
    - position
    - velocity
    - mass
    - name
    
    The acceleration of each body will be initialized as zero and once all the bodies are created,
    then the acceleration it has to be computed for each body.
    
    '''
    
    def __init__(self,position,velocity,mass, name = ''):
        
        #Position
        self.position = position
        
        #Velocity
        self.velocity = velocity
        
        #Acceleration
        self.acceleration = np.zeros((3,))
        
        #Mass
        self.mass = mass
        
        #Name
        
        self.name = name
        
    
    def Distance(self,pos_i,pos_j):
        
        '''
        This function computes the distance between two bodies.
        
        In Args:
            -pos_i:          The position (x,y,z) of one particle.
            -pos_j:          The position (x,y,z) of a second particle.
            
        Out Args:
            -d:               The distance between the two particles.
        
        
        '''
    
        x_i, y_i, z_i = pos_i
        x_j, y_j, z_j = pos_j
    
        d = np.sqrt( (x_i-x_j)**2 + (y_i-y_j)**2 + (z_i-z_j)**2 )
    
        return d
    
    def Compute_acceleration(self,Bodies,p_i):
        
        '''
        This function computes the acceleration for a single body from the gravitational force between it and the 
        bodies in the array.
        
        In Args:
            - Bodies:          An array containing the objets to be considered to calculate the acceleration.
            - Index:           Index of the body, in the array Bodies, from which acceleration is to be obtained.
            
        Out Args:
            - Acceleration:    Acceleration for the body of entered index.
        
        '''
        a = np.zeros((3,))
    
        G = 4.302e-3 #pc M_sol-1 (km/s)2
    
        epsilon = 0.1
    
        for j in range(N):
        
            p_j = Bodies[j]
        
            if p_j != p_i:
            
                r = self.Distance(p_i.position,p_j.position)
            
                direction = p_i.position - p_j.position
            
                a_i = -(G*p_j.mass*direction)/np.power((r**2 + epsilon**2),3/2)
        
                a += a_i    
            
        p_i.acceleration = a # This line updates the acceleration.    
    
    
    def Compute_position(self):
        
        self.position += self.velocity*dt + 0.5*self.acceleration*dt**2
        
    
    def Compute_velocity(self):
        
        self.velocity += self.acceleration*dt
        


# In[3]:


def Initialize(Particle_Number,Size):
        
        '''
        This function creates the bodies and sets some constants to be used in the simulation.
        
        In Args:
            - N:           Number of Bodies to be simulated.
            - L:           Size of one side of the square box that contains our particles
            
        Out Args:
            - Bodies:      Array containing the bodies to be simulated.
            - Bodies_tmp:  Array containing a copy of the bodies. This is intended to be used to do 
                           a synchronous update.
        
        '''        
        global N,L,dt,v_0,Bodies
        
        N = Particle_Number
        L = Size
        v_0 = 0
        dt = 0.1
        
        #Initialize Particles

        Bodies = np.zeros((N,),dtype=object)

        for i in range(N):
    
            pos = np.random.uniform(-L,L,3)
    
            vel = np.zeros((3,))
    
            mass = np.random.uniform(0.5,20.,1)
    
            Bodies[i] = Body(pos,vel,mass,name='Particle_{}'.format(i))
        
        #Here we compute the initial acceleration for each body
    
        for i in range(N):
            
            Bodies[i].acceleration = Bodies[i].Compute_acceleration(Bodies,Bodies[i])

        print('System initialized with %d bodies and a box of side %d '%(N,L))
        
        Bodies_tmp = Bodies
    
        return Bodies, Bodies_tmp
    
def Compute(p_i):
    
    p_i.Compute_position()
    p_i.Compute_velocity()
    p_i.Compute_acceleration(Bodies,p_i)
    
    
def Update(Bodies,Tmp):
    
    Updated = pool.map(Compute,Tmp) # This line works in parallel
        
    Bodies = Updated            
        
def particles_pos(a):
    
    '''
    Purpose:      Write the position of the bodies in x,y,z arrays to be plotted later.
    
    In Args:      
            - a:  Array that contains the bodies.
            
    Out Args:
            - x
            - y
            - z
                  This are arrays containing the position in x,y,z for each particle.
    
    '''
    
    positions = np.zeros((N,3))
    
    x = np.zeros((N,))
    y = np.zeros((N,))
    z = np.zeros((N,))
    
    for i in range(N):
        
        x[i] = a[i].position[0]
        y[i] = a[i].position[1]
        z[i] = a[i].position[2]    
    
    return x,y,z
    


# ### Plot 

# In[4]:


def Simulation():
    
    global pool
    
    Stp = input('Number of steps to simulate: ')
    
    N_bodies = int(input('Number of Bodies: '))
    
    Size_of_box = int(input('Box Size: '))
    
    Steps = int(Stp)
    
    Threads = 12
    
    B1, B2 = Initialize(N_bodies,Size_of_box)

    filenames = []
    
    #Here we ask in whichs system we are to use the correct path to save de simulation.
    
    a = 0
    
    a = input('Type M if you are in MAC.\nType W if you are in Windows\n')
    
    if a == 'M' or 'm':
        
        direct = '/Users/joaquin/Google Drive/U/7.1 Complex Systems/N-Body/Code/Snaps/V2/' #MAC
        machine = 'MAC'
        pool = ThreadPool(Threads)
        
    if a == 'W' or 'w':
        
        direct = 'C:/Users/joaqu/Documents/Google Drive/U/7.1 Complex Systems/N-Body/Code/Snaps/' #WIN
        machine = 'Windows'
        pool = ThreadPool(Threads)    
        
    print('Simulating on a %s machine with %d steps using %d Threads.\n'%(machine,Steps,Threads))
    
    #This is the actual siumlation
        
    start_time = time.time()
        
    for i in range(Steps):

        plt.clf()
        fig = plt.figure(figsize=(14,14))
        ax = fig.add_subplot(111, projection='3d')

        #n = 100

        xs,ys,zs = particles_pos(B1)[0], particles_pos(B1)[1], particles_pos(B1)[2]

        ax.scatter(xs, ys, zs, c='r', marker='o', s=7)

        Lims = 2*L
        
        ax.set_xlim3d((-Lims,Lims))
        ax.set_ylim3d((-Lims,Lims))
        ax.set_zlim3d((-Lims,Lims))
        #This Sets the pannels to be White (or Transparent)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.set_axis_off()
        ax.grid(None)
        ax.set_title('Step %d'%i,fontsize=20)
        
        #Here we add some Info to the Image
        textstr = '$N=%d$\n$L=%d$\n$v_0=%d$' % (N, Lims,0)

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)

        FILENAME = 'Data_{}.png'.format(i)

        directory = direct + FILENAME

        plt.savefig(direct + FILENAME)
        plt.close()

        filenames.append(direct + FILENAME)

        Update(B1,B2)
        
    elapsed_seconds = time.time() - start_time    
    
    elapsed_time = str(datetime.timedelta(seconds=elapsed_seconds))
    
    #Here we make a gif from the outputs
    
    
    a = input('Simulation done in %s .\nDo you wish to make a video for the simulation? (Y/N)'%elapsed_time)
    
    if a == 'Y' or a == 'y':
        
        images = []

        for filename in filenames:

            images.append(imageio.imread(filename))

        gif_name = 'movie.mov'

        imageio.mimsave(direct+gif_name, images)
        
        print('\nVideo finished and saved in %s'%direct)
        
        images = []
        
    if a == 'N' or a == 'n':
        
        print('\nSimulation finished and saved in %s'%direct)

    


# In[5]:


Simulation()

