
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

N=33


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

def convolve(x1, x2):
    conv = np.zeros(len(x1))
    for i in range(len(x2)):
        conv = conv + x2[i]*np.roll(x1,i)
    return conv

def run_histogram_filter(W, measurements, likelihood, prior = np.ones(N)/N):
    mean_list = [] 
    var_list = []
    for i in range(len(measurements)):
        posterior = update(prior, measurements[i], likelihood)
        predicted = convolve(posterior, W)
        prior = predicted
        mean, variance = get_hist_circular_mean_var(posterior)
        mean_list.append(mean)
        var_list.append(variance)
    return posterior, mean_list, var_list

def get_hist_circular_mean_var(hist):
    N = len(hist)
    idx = np.linspace(1,N,N)
    x = np.linspace(0,2*np.pi*(N-1)/N,N)
    x_cos = np.cos(x)
    x_sin = np.sin(x)
    mean_cos = (hist*x_cos).sum()
    mean_sin = (hist*x_sin).sum()
    mean_angle = np.arctan2(mean_sin,mean_cos)
    mean = N*mean_angle/(2*np.pi)
    if mean<0:
        mean = mean + N
    mean = mean + 1
    deltas = abs(idx - mean)
    #print(deltas)
    deltas[deltas>(N/2.0)] = deltas[deltas>(N/2.0)]-N
    variance = (hist*(deltas**2)).sum()
    return mean, variance

def get_random_sample(random_variable):
    pick_random = np.random.uniform()
    sum_prob = 0
    i = 0
    while (sum_prob<pick_random):
        sum_prob = sum_prob + random_variable[i][1]
        i = i + 1
    return random_variable[i-1][0]
    

def generate_sample(likelihood, W, initial_state=1, steps = 37):
    current_state = initial_state-1
    sample = []
    measurements = []
    probs = W[W>0]
    idxs = np.where([W>0])[1]
    random_step = [(idxs[i],p) for i,p in enumerate(probs)]
    
    random_step_list = []
    
    N = len(likelihood[list(likelihood.keys())[0]])
    sample_stats = [None for i in range(N)]
    for step in range(steps):
        random_variable = []
        for key in likelihood.keys():
            random_variable.append((key, likelihood[key][current_state]))
        
        measure = get_random_sample(random_variable)
        measurements.append(measure)
        sample.append([current_state, measure])
        
        if sample_stats[current_state] is None:
            sample_stats[current_state] = {}
        if measure not in sample_stats[current_state]:
            sample_stats[current_state][measure] = 0
        sample_stats[current_state][measure] = sample_stats[current_state][measure] + 1
        
        random_step_done = get_random_sample(random_step)
        random_step_list.append(random_step_done)
        current_state = (current_state + random_step_done)%N 
    
    
    steps_dict = (Counter(random_step_list))
    loc = 0
    for size, cant in steps_dict.items():
        loc = loc + size*cant
    
    steps_stats = np.array(list(Counter(random_step_list).values()))
    steps_stats = steps_stats/steps_stats.sum()
    return measurements, sample_stats, steps_stats, loc



# Default variables
W = np.zeros(N)
W[0] = 0.15
W[1] = 0.50
W[2] = 0.35 
pulse = np.array([0.2, 0.75, 0.9, 0.75, 0.2])
doors = np.array([4, 11, 26])
noise = 0.05
likelihood = get_room_observation_likelihood(N = N, doors = doors, pulse = pulse, noise = noise)