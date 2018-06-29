# N-Body Simulations

This code is a brute force approach of an N-Body simulation. This means that for each time step it computes the interaction between all bodies.

## The physics of the code

In an N-body simulation, the most important thing is how we rule the interaction between the bodies. In our case, this is ruled by the gravitational force. To update the position and the velocity of each body, we've used a leapfrog integrator. 

All the details of the physics are in the N-body.ipynb file.

### Scale of the system

Considering the number of bodies we can run, due to the high computational power required to run a simulation, the ideal scale to run the simulation will be the Solar System scale, then our distance will be masured in Astronomical Units (AU), the time in years (yr) and the mass in Solar Masses.

## Prerequisites

- Python version 2.6.x - 2.7.x (astroML does not yet support python 3.x)
- Numpy >= 1.4
- Scipy >= 0.7
- scikit-learn >= 0.10
- matplotlib >= 0.99

## How to use

The N-body.ipynb file work's like a guide to run different types of simulations please read it.
