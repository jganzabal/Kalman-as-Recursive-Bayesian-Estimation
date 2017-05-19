
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve1d
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def update(mean1, var1, mean2, var2):
    new_mean = float(var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1./(1./var1 + 1./var2)
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

def get_exponential_distribution_symetric(indexes, loc = 1, scale = 2):
    center = int(int(len(indexes)+1)/2)
    exponential_dist = np.exp(-abs(indexes-center)/scale)
    exponential_dist = exponential_dist/exponential_dist.sum()
    n = center - loc
    return np.roll(exponential_dist,-n)

def get_exponential_distribution(indexes, loc = 1, scale = 2, mirror = False):
    exponential_dist = np.exp(-abs(indexes)/scale)
    exponential_dist = exponential_dist/exponential_dist.sum()
    if mirror:
        return np.roll(exponential_dist[::-1],loc)
    return np.roll(exponential_dist,loc-1)

def plot_distribution(indexes, states_probabilities, title = ''):
    width = 1/1.5
    plt.figure(figsize=(20, 3))
    plt.bar(indexes, states_probabilities, width=width)
    plt.xticks(indexes)
    plt.title(title)
    plt.show()

def after_observation_distribution_update(states_distribution, observation_dist):
    new_distribution = states_distribution * observation_dist
    new_distribution = new_distribution/new_distribution.sum()
    return new_distribution

def stochastic_next_step_distribution_prediction(dist1, dist2):
    center = int(int(len(dist1))/2)
    return convolve1d(dist1, dist2, mode = 'wrap', origin = -center)

def deterministic_next_step_distribution_prediction(states_distribution):
    return np.roll(states_distribution,1)

def plot_robot_room(N = 21, loc = 0):
    pos_radius = 0.8
    small_radius = 0.1
    angle = 0
    N = 21
    delta_angle = 2*np.pi/N
    fig, ax = plt.subplots(figsize=(8,8))
    #plt.figure(figsize=(5,5))
    im = plt.imread('robot.png')
    oi = OffsetImage(im, zoom = 0.25)
    rx = 0
    ry = 0
    
    for i in range(N):
        x = pos_radius*np.cos(angle)
        y = pos_radius*np.sin(angle)
        if (i+1) == loc:
            box = AnnotationBbox(oi, (x, y), frameon=False)
            ax.add_artist(box)
            pos_circle = plt.Circle((x, y), radius=small_radius, fill=True, color = 'y')
        else:
            pos_circle = plt.Circle((x, y), radius=small_radius, fill=False, color = 'g')
        plt.gca().add_patch(pos_circle)
        angle = angle + delta_angle
        plt.text(x, y, str(i+1), fontsize = 12, horizontalalignment='center', verticalalignment='center')
    circle_out = plt.Circle((0, 0), radius=1, fill=False)
    plt.gca().add_patch(circle_out)
    circle_in = plt.Circle((0, 0), radius=0.95, fill=False)
    plt.gca().add_patch(circle_in)
    circle_inner = plt.Circle((0, 0), radius=0.65, fill=False)
    plt.gca().add_patch(circle_inner)

    plt.axis('scaled')
    plt.axis('off')
    plt.show()