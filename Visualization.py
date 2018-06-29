import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from Body import Body
from astroML.correlation import two_point
from scipy.optimize import curve_fit

def calc_J(Bodies):
    
    J_vec_total=np.zeros((3,))
    
    N = len(Bodies)
    
    for i in range(N): 
        
        J_vec_total+=Bodies[i].mass*np.cross(Bodies[i].position, Bodies[i].velocity)
        
    J_norm=np.linalg.norm(J_vec_total)
    
    return J_vec_total, J_norm

def Energy(Bodies):
    
    val_Energia=0
    
    G = 39.478 # Gravitational Constant in units AU^3 M_sol-1 yr^-2
    
    epsilon=0.01 #Softening Parameter
    
    N = len(Bodies)
    
    for i in range(N): 
        
        Energia_potencial=0
        
        for j in range(N):
        
            if Bodies[j].name!= Bodies[i].name:
                
                direction = Bodies[i].position - Bodies[j].position
                
                Energia_potencial+= -(G*Bodies[j].mass*Bodies[i].mass)/np.sqrt((np.linalg.norm(direction)**2+epsilon**2))   

        val_Energia+=0.5*Bodies[i].mass*np.linalg.norm(Bodies[i].velocity)**2+Energia_potencial
    
    return val_Energia

def particles_mass(a):
    
    N = len(a)
    
    masa_array = np.zeros((N,))
    
    for i in range(N):
        
        masa_array[i] = a[i].mass
  
    
    return masa_array

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
    
    N = len(a)
    
    positions = np.zeros((N,3))
    
    x = np.zeros((N,))
    y = np.zeros((N,))
    z = np.zeros((N,))
    
    for i in range(N):
        
        x[i] = a[i].position[0]
        y[i] = a[i].position[1]
        z[i] = a[i].position[2]    
    
    return x,y,z

def Plot(Bodies,step,Nplot,L,direct = '',Trajectory = [False,0]):
    
    '''
         Saves a plot of the actual position of the bodies.
    
    In Args:      
            - Bodies:      Array that contains the bodies.
            - step:        Actual step of the simulation.
            - Nplot:       Number of snapshot.
            - L:           Scale of the plot.
            - direct:      Directory where the plots will be saved.
            - Trajectory:  -- First entry defines if you want to trace an object's trajectory
                           -- Second entry defines the object to be traced (By default 0).
            
    Out Args:
            - None
    
    '''
    
    
    plt.clf()
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111, projection='3d')

    xs,ys,zs = particles_pos(Bodies)
    
    mass_color = particles_mass(Bodies)

    if Trajectory[0] == True:
        
        try:
            
            Object = Trajectory[1]
            
            trayectory_x.append(xs[Object])
            trayectory_y.append(ys[Object])
            trayectory_z.append(zs[Object])
        
            ax.plot(trayectoria_x, trayectoria_y, trayectoria_z)
            
        except IndexError: #This will handle when the object index to be traced is outside the array of Bodies.
            
            print('Particle to be traced does not exist.')
    
    p = ax.scatter(xs, ys, zs, marker='o', c=mass_color, cmap = cm.coolwarm)
    clb = fig.colorbar(p,)
    clb.set_label(r'$M_\odot$', fontsize = 25, rotation= 0 )

    Lims = 2*L
        
    ax.set_xlim3d((-Lims,Lims))
    ax.set_ylim3d((-Lims,Lims))
    ax.set_zlim3d((-Lims,Lims))
    
    # This Sets the pannels to be White (or Transparent)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off() # This eliminates the axis
    ax.grid(None)
    
    ax.set_title('Snap %d'%Nplot,fontsize=20)
        
    # Here we add some Info to the Image
    vector1,val2 = calc_J(Bodies)
    val3 = Energy(Bodies)
    
    textstr1 = r'Step=$%d$' % (step)
    textstr2 = r'$\vec{J}$=$(%.2f,%.2f,%.2f)$' % (vector1[0],vector1[1],vector1[2])
    textstr3 = r'$||\vec{J}||$=$%.2f$' % (val2)
    textstr4 = r'$E_{total}$=$%.2f$' % (val3)
 
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text2D(0.05, 0.95, textstr1, transform=ax.transAxes, fontsize=18,verticalalignment='top', bbox=props)
    ax.text2D(0.05, 0.90, textstr2, transform=ax.transAxes, fontsize=18,verticalalignment='top', bbox=props)
    ax.text2D(0.05, 0.85, textstr3, transform=ax.transAxes, fontsize=18,verticalalignment='top', bbox=props)
    ax.text2D(0.05, 0.80, textstr4, transform=ax.transAxes, fontsize=18,verticalalignment='top', bbox=props)
    
    FILENAME = 'Data{}.png'.format(Nplot)

    plt.savefig(direct + FILENAME)        
    plt.close()

def PlotCorr(DATA,direct = '',axlims = [0,4e0,8e-2,1.2e1],show=False): # This plots the Correlation Funtion from DATA already computed
                                                                       # with ComputeCorr. 
    
    def Corr_fit(r,a,b):
    
        return a * np.power(r,b)
     
    x_data = DATA[:,0]
    y_data = DATA[:,1]
    
    popt, pcov = curve_fit(Corr_fit,x_data,y_data) # popt are the a,b fitted constants for the power-law
                                                   # pcov is the covariance matrix for those constants
    
    perr = np.sqrt(np.diag(pcov)) # perr are the errors for the constants fitted.
    
    # Data for the plot

    x_fit = np.linspace(3e-4,500,1000)

    y_fit = Corr_fit(x_fit,popt[0], popt[1])
    y_sup = Corr_fit(x_fit,popt[0] + perr[0], popt[1] + perr[1])
    y_inf = Corr_fit(x_fit,popt[0] - perr[0], popt[1] - perr[1])
    
    # Here we make the Plot
    
    plt.figure(figsize=(15,10))
    ax = plt.subplot(111)

    #Data
    plt.scatter(x_data,y_data,s=20,label='Data',zorder=2)

    #Fit
    plt.loglog(x_fit,y_fit,color='orange',lw=3,zorder=1,label = r'Best Fit ($\xi(r) + 1 = %.2f r^{%.2f}$)'%(popt[0], popt[1]))

    #Errors
    plt.fill_between(x_fit,y_fit , y_sup, facecolor='lightgray', interpolate=True)
    plt.fill_between(x_fit,y_inf , y_fit, facecolor='lightgray', interpolate=True)

    # Plot Parameters
    plt.axis(axlims)
    plt.xlabel(r'$r$  in Astronomical Units',fontsize=25)
    plt.ylabel(r'Correlation Function [$\xi(r) + 1$]', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=25)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tick_params(axis='both',which='both',direction='in', length=6, width=1)
    
    plt.savefig(direct + 'CorrPlot.png')
    
    if show == True:
        plt.show()
    plt.close()

def ComputeCorr(Bodies,direct = '', nbins = 250 ): #This Computes the correlation function for the bodies in the array Bodies.
    
    X = np.zeros((len(Bodies),3))

    for i in range(len(Bodies)):

        X[i] = Bodies[i].position
    
    bins = np.logspace(-5, 2,int(nbins))
    
    corr = two_point(X, bins) + 1 #This is where the two point correlation is computed and stored in an array named corr.
    
    DATA = np.array([bins[1:],corr]).T # This array contains the correlation function data as (r,corr) points.

    DATA = DATA[~np.any(np.isnan(DATA), axis=1)] # Removes nan's because curve_fit doesn't handle them well.

    np.savetxt(direct + 'Corr.txt', DATA, fmt='%f')

    # a = np.loadtxt('Corr.txt') # To load the data form text
    
    return DATA
    
