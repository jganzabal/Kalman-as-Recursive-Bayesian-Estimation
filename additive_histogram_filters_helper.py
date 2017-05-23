import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

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

def update(p, X, likelihood):
    # p: prior probability
    # X: Measurement. Measured position
    # posterior not normalized
    posterior = likelihood[X]*p
    # Normalize it
    normalized = posterior/posterior.sum()
    return normalized

def convolve(x1, x2):
    conv = np.zeros(len(x1))
    for i in range(len(x2)):
        conv = conv + x2[i]*np.roll(x1, i)
    return conv

def get_random_sample(random_variable):
    pick_random = np.random.uniform()
    sum_prob = 0
    i = 0
    while sum_prob < pick_random:
        sum_prob = sum_prob + random_variable[i][1]
        i = i + 1
    return random_variable[i-1][0]

def generate_sample(likelihood, W, initial_state=1, steps = 37):
    current_state = initial_state-1
    sample = []
    measurements = []
    probs = W[W > 0]
    idxs = np.where([W > 0])[1]
    random_step = [(idxs[i], p) for i, p in enumerate(probs)]

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
        current_state = (current_state + random_step_done) % N 

    steps_dict = (Counter(random_step_list))
    loc = 1
    for size, cant in steps_dict.items():
        loc = loc + size*cant
    loc = loc%N
    print(steps_dict)
    steps_stats = np.array(list(Counter(random_step_list).values()))
    steps_stats = steps_stats/steps_stats.sum()
    return measurements, sample_stats, steps_stats, loc

def get_hist_circular_mean_var(hist):
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
            while abs(mean_list[-1]-mean) > N/2:
                mean = mean + N_mult*N

        if len(mean_list_pred) > 1:
            while abs(mean_list_pred[-1]-mean_pred) > N/2:
                mean_pred = mean_pred + N_mult_pred*N

        mean_list.append(mean)
        var_list.append(variance)
        mean_list_pred.append(mean_pred)
        var_list_pred.append(variance_pred)
    return posterior, mean_list, var_list, mean_list_pred, var_list_pred

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