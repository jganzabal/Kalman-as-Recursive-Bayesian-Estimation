from numpy.random import uniform, seed
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib import pyplot as plt

def gauss_n_dimensional(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_countour(x,y,z, limits, color = 'r'):
    # define grid.
    xi = np.linspace(limits[0], limits[1], 100)
    yi = np.linspace(limits[2], limits[3], 100)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.85, 0.9, 0.95]
    n_levels = len(levels)
    #n_levels = 10
    # contour the gridded data, plotting dots at the randomly spaced data points.
    #CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    #CS = plt.contourf(xi,yi,zi,n_levels,cmap=cm.Greys_r, levels=levels)
    
    #CS = plt.contourf(xi,yi,zi,n_levels,cmap=cm.Greys_r)
    CS = plt.contour(xi,yi,zi,n_levels,linewidths=0.5,colors=color, levels=levels)
    
    #plt.colorbar() # draw colorbar
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)
    
def plot_2d_gauss(mu ,Sigma ,npts = 1000, color='r',limits = [-1,2, -1.5,2.5]):
    x = uniform(limits[0], limits[1], npts)
    y = uniform(limits[2], limits[3], npts)
    z = gauss_n_dimensional(x,y,Sigma,mu)
    plot_countour(x,y, z, limits = limits, color = color)