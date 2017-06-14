import numpy as np
from collections import Counter
from matplotlib import pyplot as plt


## Room observation likelihood
def get_room_observation_likelihood(N=33, doors=np.array([4, 11, 26]),\
                                    pulse=np.array([0.2, 0.8, 1, 0.8, 0.2]), noise=0.05):
    uniform = np.zeros(N)
    uniform[doors-1] = 1
    unnormalized = np.roll(convolve(uniform, pulse), -int((len(pulse)-1)/2))
    unnormalized[unnormalized == 0] = noise
    likelihood = {}
    likelihood['door'] = unnormalized
    likelihood['wall'] = 1-unnormalized
    return likelihood

## Additive noise functions
def get_gaussian_distribution(N=21, loc=1, scale=2):
    center = int(int(N+1)/2)
    x = np.linspace(1, N, N)
    x = x - center
    sigma = scale*2
    pdf = np.exp(-x*x/(2*sigma))
    n = center - loc
    return np.roll(pdf/pdf.sum(), -n)

def get_rayleigh_distribution(N=21, loc=1, scale=2):
    center = int(int(N+1)/2)
    x = np.linspace(1, N, N)
    sigma = scale*4
    x = x - center + np.sqrt(sigma)
    pdf = (x >= 0)*(x*np.exp(-x*x/(2*sigma))/sigma)
    n = center - loc
    return np.roll(pdf/pdf.sum(), -n)

def get_exponential_distribution_symetric(N=21, loc=1, scale=2):
    indexes = np.linspace(1, N, N)
    center = int(int(len(indexes)+1)/2)
    exponential_dist = np.exp(-abs(indexes-center)/scale)
    exponential_dist = exponential_dist/exponential_dist.sum()
    n = center - loc
    return np.roll(exponential_dist, -n)

def get_exponential_distribution(N=21, loc=1, scale=2, mirror=False):
    indexes = np.linspace(1, N, N)
    exponential_dist = np.exp(-abs(indexes)/scale)
    exponential_dist = exponential_dist/exponential_dist.sum()
    if mirror:
        return np.roll(exponential_dist[::-1], loc)
    return np.roll(exponential_dist, loc-1)

## Walking functions
def get_walking_noise_example_1(N):
    W = np.zeros(N)
    W[0] = 0.15
    W[1] = 0.50
    W[2] = 0.35
    return W

def get_walking_noise_example_2(N):
    W = np.zeros(N)
    W[0] = 0.15
    W[1] = 0.70
    W[2] = 0.15
    return W

def get_walking_noise_perfect_1(N):
    W = np.zeros(N)
    W[1] = 1
    return W

def get_walking_noise_perfect_3(N):
    W = np.zeros(N)
    W[3] = 1
    return W

## Generative functions
def get_random_sample(random_variable):
    pick_random = np.random.uniform()
    sum_prob = 0
    i = 0
    while sum_prob < pick_random:
        sum_prob = sum_prob + random_variable[i][1]
        i = i + 1
    return random_variable[i-1][0]


def generate_sample(likelihood, W, initial_state=1, steps=37):
    current_state = initial_state-1
    sample = []
    measurements = []
    probs = W[W > 0]
    idxs = np.where([W > 0])[1]
    random_step = [(idxs[i], p) for i, p in enumerate(probs)]

    random_step_list = []
    real_locations = []
    N = len(likelihood[list(likelihood.keys())[0]])
    sample_stats = [None for i in range(N)]
    for step in range(steps):
        random_variable = []
        for key in likelihood.keys():
            random_variable.append((key, likelihood[key][current_state]))

        measure = get_random_sample(random_variable)
        measurements.append(measure)
        sample.append([current_state, measure])
        real_locations.append(current_state)
        if sample_stats[current_state] is None:
            sample_stats[current_state] = {}
        if measure not in sample_stats[current_state]:
            sample_stats[current_state][measure] = 0
        sample_stats[current_state][measure] = sample_stats[current_state][measure] + 1

        random_step_done = get_random_sample(random_step)
        random_step_list.append(random_step_done)
        current_state = (current_state + random_step_done) % N
    
    transition_hist = Counter(random_step_list)
    steps_stats = []
    for i in range(len(W)):
        if i in transition_hist:
            steps_stats.append(transition_hist[i]/steps)
        else:
            steps_stats.append(0)
    #steps_stats = [transition_hist[i]/steps for i in range(len(transition_hist))]
    return measurements, sample_stats, steps_stats, real_locations

## Histogram filter functions

def convolve(x1, x2):
    conv = np.zeros(len(x1))
    for i in range(len(x2)):
        conv = conv + x2[i]*np.roll(x1, i)
    return conv

def update(p, X, likelihood):
    # p: prior probability
    # X: Measurement. Measured position
    # posterior not normalized
    posterior = likelihood[X]*p
    # Normalize it
    normalized = posterior/posterior.sum()
    return normalized

def prediction(posterior, transition):
    # posterior: posterior probability distribution
    # transition: transition probability distribution
    return convolve(posterior, transition)

def get_hist_circular_mean_var(hist, zero_centered=True):
    N = len(hist)
    idx = np.linspace(1, N, N)
    x = np.linspace(0, 2*np.pi*(N-1)/N, N)
    x_cos = np.cos(x)
    x_sin = np.sin(x)
    mean_cos = (hist*x_cos).sum()
    mean_sin = (hist*x_sin).sum()
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean = N*mean_angle/(2*np.pi)
    if mean < 0:
        mean = mean + N
    mean = mean + 1
    deltas = abs(idx - mean)
    deltas[deltas > (N/2.0)] = deltas[deltas > (N/2.0)]-N
    variance = (hist*(deltas**2)).sum()
    if zero_centered:
        if mean > N/2:
            mean = mean - N
    return mean, variance

def run_histogram_filter(W, measurements, likelihood, prior):
    N = len(prior)
    mean, variance = get_hist_circular_mean_var(prior)
    mean_list = [mean]
    var_list = [variance]
    mean_list_pred = [mean]
    var_list_pred = [variance]
    N_mult = 1
    N_mult_pred = 1
    for i in range(len(measurements)):
        posterior = update(prior, measurements[i], likelihood)
        mean, variance = get_hist_circular_mean_var(posterior)
        predicted = convolve(posterior, W)
        prior = predicted
        mean_pred, variance_pred = get_hist_circular_mean_var(prior)

        if len(mean_list) > 1:
            while abs(mean_list[-1]-mean) > (N/2):
                #print(abs(mean_list[-1]-mean), mean)
                mean = mean + N_mult*N

        if len(mean_list_pred) > 1:
            while abs(mean_list_pred[-1]-mean_pred) > (N/2):
                mean_pred = mean_pred + N_mult_pred*N

        mean_list.append(mean)
        var_list.append(variance)
        mean_list_pred.append(mean_pred)
        var_list_pred.append(variance_pred)
    return posterior, mean_list, var_list, mean_list_pred, var_list_pred

## Ploting functions
def plot_estimations(mean_list, var_list, mean_list_pred, var_list_pred, fr=0, to=-1):
    if to < 0:
        to = len(mean_list)
    plt.plot(mean_list[fr:to], color='b')
    mean_list_pred_1 = np.array(mean_list_pred)-1
    plt.plot((mean_list_pred_1)[fr:to], color='r')
    plt.plot((mean_list+np.sqrt(var_list))[fr:to], color='g')
    plt.plot((mean_list-np.sqrt(var_list))[fr:to], color='g')
    plt.plot((mean_list_pred_1+np.sqrt(var_list_pred))[fr:to], color='y')
    plt.plot((mean_list_pred_1-np.sqrt(var_list_pred))[fr:to], color='y')
    plt.show()

def plot_distribution(data, title='', fig=None, color='b', str_indexes=None, rotation=0, mark=None):
    N = len(data)
    indexes = np.linspace(1, N, N)
    if fig is None:
        fig, ax = plt.subplots(figsize=(20, 3))
    width = 1/1.5
    plt.bar(indexes, data, width=width, color=color)
    plt.xticks(rotation=rotation)
    if not str_indexes == -1:
        if str_indexes is None:
            plt.xticks(indexes)
        else:
            plt.xticks(indexes, str_indexes)
    plt.title(title)
    if mark is not None:
        plt.bar(mark, data[mark-1], width=width, color='r')

from ipywidgets import *
from scipy.stats import entropy

def histogram_filter(W, measurements, likelihood, prior):
    # W: transition probability distribution
    # measurements: It is a list of observations. The i'th observation Xi = measurements[i]
    # likelihood: It is a dict where likelihood[Xi] is the likelihood given observation Xi
    # prior: The initial distribution, normaly with normalized entropy of 1 (Maximun confusion)
    normalized_entropy = []
    mean_array = []
    var_array = []
    for i in range(len(measurements)):
        posterior = update(prior, measurements[i], likelihood)
        normalized_entropy.append(entropy(posterior, base=2)/np.log2(len(prior)))
        mean, variance = get_hist_circular_mean_var(posterior)
        mean_array.append(mean)
        var_array.append(variance)
        predicted = prediction(posterior, W)
        prior = predicted
    return posterior, predicted, normalized_entropy, mean_array, var_array

import matplotlib.patches as mpatches
def plot_histogram_entropy_std(measurements, transition, likelihood, prior, n_steps=1, real_positions=None):    
    N = len(prior)
    posterior, predicted, normalized_entropy, mean_array, var_array\
                     = histogram_filter(transition,
                                        measurements[:n_steps],
                                        likelihood,
                                        prior=np.ones(N)/N)
    f = plt.figure(figsize=(20, 10))
    plt.subplot(3, 1, 1)
    plot_distribution(posterior, title='normalized $P(S=k|X)$ - Posterior -', fig=f)
    #plt.show()
    #plt.figure(figsize=(20,5))
    plt.subplot(3, 2, 3)
    plt.title("Normalized entropy")
    plt.plot(normalized_entropy)
    plt.subplot(3, 2, 4)
    plt.title("Standard deviation")
    plt.plot(np.array(var_array)**(0.5))
    if (real_positions is not None) and (type(measurements[0]) is not int):
        real_positions = real_positions[:n_steps]
        plt.subplot(3, 1, 3)
        measurements_options = list(set(measurements[:n_steps]))
        color = ['r','b','g','y','k']
        map_dict = {}
        for i,meas in enumerate(measurements_options):
            map_dict[meas] = color[i]
        for i, measurement in enumerate(measurements[:n_steps]):
            plt.scatter(i, real_positions[i], color=map_dict[measurement], label=measurement)
        plt.plot(real_positions)
        class_colours = color[:len(measurements_options)]
        recs = []
        for i in range(0,len(class_colours)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
        plt.legend(recs,measurements_options,loc=4)
        plt.xlabel('iteration')
        plt.ylabel('Robot position')
    elif (real_positions is not None):
        plt.subplot(3, 1, 3)
        plt.plot(measurements[:n_steps], label="Robot location measured")
        plt.plot(real_positions[:n_steps], label="Robot real location")
        plt.xlabel('iteration')
        plt.ylabel('Robot position')
        plt.legend()

    plt.show()
    print("normalized entropy of last posterior:", normalized_entropy[-1])


def plot_interactive_histogram(measurements, transition, likelihood, prior, steps, real_locations=None, initial_slider_pos=10):
    plot_histogram_result_interactive = lambda n_steps=initial_slider_pos: plot_histogram_entropy_std(measurements, transition, 
                                                                                          likelihood, prior,
                                                                                          n_steps, real_positions = real_locations) 

    interact(plot_histogram_result_interactive, n_steps = widgets.IntSlider(min=1, max=steps,
                                                                            step=1, value=initial_slider_pos,
                                                                            continuous_update=False))


def get_likelihood(N, observation_func, scale=6):
    likelihood = {}
    for X_0 in range(N):
        X = X_0 + 1
        likelihood[X] = []
        for k in range(N):
            likelihood_k = observation_func(N= N, loc = k+1, scale=scale)
            likelihood[X].append(likelihood_k[X-1])
    return likelihood
