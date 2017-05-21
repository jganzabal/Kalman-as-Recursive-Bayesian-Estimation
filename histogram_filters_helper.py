
import numpy as np
from matplotlib import pyplot as plt

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

def update(p, X, likelihood):
    # p: prior probability
    # X: Measurement, can be door or wall
    # posterior not normalized
    posterior = likelihood[X]*p
    # Normalize it
    normalized = posterior/posterior.sum()
    return normalized