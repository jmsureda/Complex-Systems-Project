import numpy as np
import matplotlib.pylab as plt
from astroML.correlation import two_point
from scipy.optimize import curve_fit


def PlotCorr(DATA,direct = '',axlims = [0,4e0,8e-2,1.2e1],show=False):
    
    def Corr_fit(r,a,b):
    
        return a * np.power(r,b)
     
    x_data = DATA[:,0]
    y_data = DATA[:,1]
    
    popt, pcov = curve_fit(Corr_fit,x_data,y_data)

    perr = np.sqrt(np.diag(pcov))
    
    # Data for the plot

    x_fit = np.linspace(3e-3,50,100)

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


# In[ ]:


def ComputeCorr(Bodies,direct = '', nbins = 250 ):
    
    X = np.zeros((len(Bodies),3))

    for i in range(len(Bodies)):

        X[i] = Bodies[i].position
    
    bins = np.logspace(-5, 2,int(nbins))
    
    corr = two_point(X, bins) + 1
    
    DATA = np.array([bins[1:],corr]).T

    #remove nan's
    DATA = DATA[~np.any(np.isnan(DATA), axis=1)]

    np.savetxt(direct + 'Corr.txt', DATA, fmt='%f')

    # a = np.loadtxt('Corr.txt') # To load the data form text
    
    return DATA
    

