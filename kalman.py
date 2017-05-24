from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt


def gaussian(x, mu, sig):
    #return norm.pdf(x, loc = mu, scale = sig)
    return np.exp(-np.power((x - mu)/sig, 2.)/2)/(np.sqrt(2.*np.pi)*sig)

def plot_gaussian(mu=0, sig= 1, points = 200, N = 2, marker=None, label = None, x = None, color = None, 
                  x_label='',y_label='',ax = None, figsize = (20,10), center_label_tick = None):
    if ax is None:
        f, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize = figsize)
    if x is None:
        x = np.linspace(mu-N*sig, mu+N*sig, points)
    y = gaussian(x, mu, sig)
    ax.plot(x,y, marker = marker, label = label, color = color)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if center_label_tick is not None:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[int((len(labels)-1)/2)] = center_label_tick
        ax.set_xticklabels(labels)

    return x, y

def update(mean1, var1, mean2, var2):
    new_mean = float(var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1./(1./var1 + 1./var2)
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

def plot_kalman_process(mu, sig, measurement_sig, motion_sig, points, measurements, motion):
    x = np.linspace(mu-2*10, mu+2*10, points)
    rows = int(np.ceil(len(measurements)/3))
    f, ax = plt.subplots(rows, 3, sharey=True, sharex=True, figsize = (20,10))
    ax = ax.flatten()
    for n in range(len(measurements)):
        label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(mu,sig)
        plot_gaussian(mu=mu, sig= sig, points = points, N = 2, x=x, label = '(Initial) '+label_data, color = 'b', ax=ax[n])
        [mu, sig] = update(mu, sig, measurements[n], measurement_sig)
        label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(mu,sig)
        plot_gaussian(mu=mu, sig= sig, points = points, N = 2, x=x, label = '(Update) '+label_data, color = 'r', ax=ax[n])
        #print('update:',[mu, sig])
        [mu, sig] = predict(mu, sig, motion[n], motion_sig)
        label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(mu,sig)
        plot_gaussian(mu=mu, sig= sig, points = points, N = 2, x=x, label = '(Predict) '+label_data, color = 'y', ax=ax[n])
        #print('predict:',[mu, sig])
        ax[n].legend()
    plt.show()