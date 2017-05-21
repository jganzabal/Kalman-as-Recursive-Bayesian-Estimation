
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

def plot_distribution(data, title = '', fig = None, color= 'b', str_indexes = None, rotation = 0):
    N = len(data)
    indexes = np.linspace(1,N,N)
    if fig is None:
        fig, ax = plt.subplots(figsize=(20, 3))
    width = 1/1.5
    plt.bar(indexes, data, width=width, color= color)
    plt.xticks(rotation=rotation)
    if str_indexes is None:
        plt.xticks(indexes)
    else:
        plt.xticks(indexes, str_indexes)
    plt.title(title)

def plot_iteration(indexes, action, observed, prior, observation_not_mirr, update, prediction, fig = None, color= 'b'):
    if fig is None:
        fig, ax = plt.subplots(3,2,figsize=(20, 3))
    ax = fig.axes
    width = 1/1.5
    fig.sharex = True
    plt.xticks(indexes)
    ax[0].bar(indexes, action, width=width, color= color)
    ax[0].set_title("Action")

    ax[1].bar(indexes, observed, width=width, color= color)
    ax[1].set_title("Observed mirrored")

    ax[2].bar(indexes, prior, width=width, color= color)
    ax[2].set_title("Prior")

    ax[3].bar(indexes, observation_not_mirr, width=width, color= color)
    #ax[1].set_xticks(indexes)
    ax[3].set_title("Observation")

    ax[4].bar(indexes, update, width=width, color= color)
    #ax[2].set_xticks(indexes)
    ax[4].set_title("Update")

    ax[5].bar(indexes, prediction, width=width, color= color)
    #ax[3].set_xticks(indexes)
    ax[5].set_title("Prediction")

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

# Histogram filters

def convolve(x1, x2):
    conv = np.zeros(len(x1))
    for i in range(len(x2)):
        conv = conv + x2[i]*np.roll(x1,i)
    return conv

def get_room_observation_likelihood(N = 33, doors = np.array([4, 11, 26]), pulse = np.array([0.2, 0.8, 1, 0.8, 0.2]), noise = 0.05):
    uniform = np.zeros(N)
    uniform[doors-1] = 1
    unnormalized = np.roll(convolve(uniform, pulse),-int((len(pulse)-1)/2))
    unnormalized[unnormalized==0] = noise
    likelihood = {}
    likelihood['door'] = unnormalized
    likelihood['wall'] = 1-unnormalized
    return likelihood