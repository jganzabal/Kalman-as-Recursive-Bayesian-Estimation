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

def gauss_pdf_mult(mean1, var1, mean2, var2):
    new_mean = float(var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1./(1./var1 + 1./var2)
    return [new_mean, new_var]

def update(h, sigma_v, Z, X_est_prior ,P_prior):
    P = P_prior*sigma_v/(P_prior*(h**2) + sigma_v)
    X_est = P*(h*Z/sigma_v + X_est_prior/P_prior)
    return X_est, P

def gauss_var_add(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

def predict(sigma_w, X_updated, P_updated, a = 1, b = 1, U = 1):
    X_predicted = a*X_updated + b*U
    P_predicted = (a**2)*sigma_w + P_updated
    return X_predicted, P_predicted

def plot_kalman_process(measurements, X_est_prior, P_prior, sigma_v, sigma_w, points = 200, h = 1, a = 1, b = 1, U = 1):
    x = np.linspace(X_est_prior-2*2, X_est_prior+2*10, points)
    rows = int(np.ceil(len(measurements)/3))
    f, ax = plt.subplots(rows, 3, sharey=True, sharex=True, figsize = (20,10))
    ax = ax.flatten()
    for n in range(len(measurements)):
        label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_est_prior,P_prior)
        plot_gaussian(mu=X_est_prior, sig= P_prior, points = points, N = 2, x=x, label = '(Initial) '+label_data, color = 'b', ax=ax[n])
        Z = measurements[n]
        X_updated, P_updated = update(h, sigma_v, Z, X_est_prior ,P_prior)
        label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_updated,P_updated)
        plot_gaussian(mu=X_updated, sig= P_updated, points = points, N = 2, x=x, label = '(Update) '+label_data, color = 'r', ax=ax[n])
        
        X_predicted, P_predicted = predict(sigma_w, X_updated, P_updated, a = a, b = b, U = U)
        label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_predicted,P_predicted)
        plot_gaussian(mu=X_predicted, sig= P_predicted, points = points, N = 2, x=x, label = '(Predict) '+label_data, color = 'y', ax=ax[n])
        #print('predict:',[mu, sig])
        ax[n].legend()
        X_est_prior = X_predicted
        P_prior = P_predicted
    plt.show()