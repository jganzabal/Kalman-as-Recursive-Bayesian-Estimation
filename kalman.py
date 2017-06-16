from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import *

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

def gauss_var_add(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]


def update(h, sigma_v, Z, X_est_prior ,P_prior):
    P = P_prior*sigma_v/(P_prior*(h**2) + sigma_v)
    X_est = P*(h*Z/sigma_v + X_est_prior/P_prior)
    return X_est, P

def predict(sigma_w, X_updated, P_updated, a = 1, b = 1, U = 1):
    X_predicted = a*X_updated + b*U
    P_predicted = (a**2)*sigma_w + P_updated
    return X_predicted, P_predicted



def generate_sample(X_o = 0, sigma_w = 0.1,sigma_v = 0.1, h = 1, a = 1, b = 1, U = 1, steps = 10):
    real_positions = [X_o]
    measurements = [h*X_o + np.random.normal(0,sigma_v)]
    X_k = X_o
    for i in range(steps):
        X_k = a*X_k + b*U + np.random.normal(0,sigma_w)
        real_positions.append(X_k)
        Z = h*X_k + np.random.normal(0,sigma_v)
        measurements.append(Z)
    return measurements, real_positions

def kalman_filter_(measurements, X_est_prior, P_prior, sigma_v, sigma_w, h = 1, a = 1, b = 1, U = 1, predict_ratio = 1):
    updated_means = [X_est_prior]
    update_variances = [P_prior]
    predicted_means = [X_est_prior]
    predicted_variances = [P_prior]
    for n in range(len(measurements)):
        Z = measurements[n]
        if n%predict_ratio == 0:
            X_updated, P_updated = update(h, sigma_v, Z, X_est_prior ,P_prior)
        else:
            X_updated, P_updated = X_predicted, P_predicted
        
        updated_means.append(X_updated)
        update_variances.append(P_updated)
            
        X_predicted, P_predicted = predict(sigma_w, X_updated, P_updated, a = a, b = b, U = U)
        predicted_means.append(X_predicted)
        predicted_variances.append(P_predicted)
        
        X_est_prior = X_predicted
        P_prior = P_predicted
    return updated_means, update_variances, predicted_means, predicted_variances

def get_asyntotic_params(sig_v, sig_w, a = 1):
    P_pred = (sig_w + sig_v*(a**2-1) +  np.sqrt((sig_w + sig_v*(a**2-1))**2 + 4*sig_w*sig_v))/2
    P_obs = P_pred - sig_w
    K = P_pred/(P_pred + sig_v)
    p_n1 = (1-K)*P_pred + sig_w
    print(P_pred, P_obs, K, p_n1)


def plot_kalman_process(measurements, X_est_prior, P_prior, sigma_v, sigma_w, real_positions=None, points = 200, h = 1, a = 1, b = 1, U = 1):
    rows = int(np.ceil(len(measurements)/3))
    f, ax = plt.subplots(rows, 3, sharey=True, sharex=True, figsize = (20,10))
    ax = ax.flatten()
    x_min = min(measurements)
    x_max = max(measurements)
    for n in range(len(measurements)):
        Z = measurements[n]
        actual_position = real_positions[n]
        X_updated, P_updated = update(h, sigma_v, Z, X_est_prior ,P_prior)
        
        X_predicted, P_predicted = predict(sigma_w, X_updated, P_updated, a = a, b = b, U = U)
        
        plot_filter_densities(ax[n], X_est_prior, P_prior, X_updated, P_updated,X_predicted, P_predicted, 
                        Z = Z, actual_position=actual_position, points = points, x_limits = [x_min, x_max])

        X_est_prior = X_predicted
        P_prior = P_predicted
    plt.show()


def plot_filter_densities(ax, X_est_prior, P_prior, X_updated, P_updated,X_predicted, P_predicted, Z = None, actual_position = None, points = 200, x_limits = None, N_stds = 2):
    if x_limits is None:
        X_array = np.array([X_est_prior, X_updated, X_predicted])
        p_array = np.array([P_prior, P_updated, P_predicted])
        X_min_index = np.argmin(X_array)
        X_max_index = np.argmax(X_array)

        x = np.linspace(X_array[X_min_index]-N_stds*(p_array[X_min_index]**0.5), 
                        X_array[X_max_index]+N_stds*(p_array[X_max_index]**0.5), points)
    else:
        x = np.linspace(x_limits[0], x_limits[1], points)

    label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_est_prior,P_prior)
    plot_gaussian(mu=X_est_prior, sig= P_prior, points = points, N = 2, x=x, label = '(Prior) '+label_data, color = 'k', ax=ax)

    label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_updated,P_updated)
    plot_gaussian(mu=X_updated, sig= P_updated, points = points, N = 2, x=x, label = '(Update) '+label_data, color = 'b', ax=ax)
    
    label_data = '$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_predicted,P_predicted)
    plot_gaussian(mu=X_predicted, sig= P_predicted, points = points, N = 2, x=x, label = '(Predict) '+label_data, color = 'y', ax=ax)

    if Z is not None:
        ax.scatter(Z, 0, s=100, color="r", alpha=0.5, label = 'measurement. Z=%.2f'%Z)

    if actual_position is not None:
        ax.scatter(actual_position, 0, s=100, color="g", alpha=0.5, label = 'actual position=%.2f'%actual_position)

    ax.legend()
    

def plot_kalman_filter_results(updated_means, predicted_means, measurements, real_positions, update_variances=None, predicted_variances=None):
    plt.plot(updated_means, color = 'b', label = 'updated after observation')
    plt.plot(predicted_means, color = 'y', label = 'predicted')
    if update_variances is not None:
        plt.plot(updated_means+ 1*np.array(update_variances), color = 'k', ls='dashdot')
        plt.plot(updated_means- 1*np.array(update_variances), color = 'k', ls='dashdot')
    
    if predicted_variances is not None:
        plt.plot(predicted_means+ 1*np.array(predicted_variances), color = 'k', ls='dotted')
        plt.plot(predicted_means- 1*np.array(predicted_variances), color = 'k', ls='dotted')

    plt.plot(measurements, color = 'r', label = 'measurements')
    plt.plot(real_positions, color = 'g', label = 'real positions')
    plt.legend()

def plot_filter_densitiy_mean_std(n_steps, measurements, X_o, P_0, sigma_v, sigma_w, h = 1, a = 1, b = 1, U = 1,\
                                 x_limits = None, points = 200, N_stds=3, real_positions=None):  
    measurements_limited = measurements[:n_steps]
    real_positions_limited = real_positions[:n_steps]
    updated_means, update_variances, predicted_means, predicted_variances = kalman_filter(measurements_limited, 
                                                                                      X_o, P_0, 
                                                                                      sigma_v, 
                                                                                      sigma_w, h = 1, a = 1, 
                                                                                      b = 1, U = U)
    
    f = plt.figure(figsize=(20, 10))
    ax = plt.subplot(2, 1, 1)
    X_est_prior = predicted_means[-2]
    P_prior = predicted_variances[-2]
    X_updated = updated_means[-1]
    P_updated = update_variances[-1]
    X_predicted = predicted_means[-1]
    P_predicted = predicted_variances[-1]
    actual_position = None
    if real_positions is not None:
        actual_position = real_positions_limited[-1]

    plot_filter_densities(ax, X_est_prior, P_prior, X_updated, P_updated,X_predicted, P_predicted, Z=measurements_limited[-1], 
                            actual_position = actual_position, points = points, N_stds = N_stds, 
                            x_limits = x_limits)

    plt.subplot(2, 1, 2)
    plot_kalman_filter_results(updated_means, predicted_means, measurements_limited, real_positions_limited)
  
    plt.show()


def plot_interactive_kalman_filter(measurements, X_o, P_0, sigma_v, sigma_w, steps, h = 1, a = 1, b = 1,U = 1,\
                                    N_stds=3, real_positions=None, x_limits = None, initial_slider_pos=5):
    # steps is the max number of steps
    if x_limits is None:
        x_limits = [min(measurements[:steps]), max(measurements[:steps])]
    plot_interactive_kalman_filter_result =lambda n_steps= initial_slider_pos:plot_filter_densitiy_mean_std(n_steps, measurements, X_o, P_0, sigma_v, 
                                        sigma_w, h = h, a = a, b = b, U = U, N_stds=N_stds, real_positions=real_positions, x_limits=x_limits)


    interact(plot_interactive_kalman_filter_result, n_steps = widgets.IntSlider(min=1, max=steps,
                                                                            step=1, value=initial_slider_pos,
                                                                            continuous_update=False))


def kalman_filter(measurements, X_est_prior, P_prior, sigma_v, sigma_w, h = 1, a = 1, b = 1, U = 1):
    updated_means = []
    update_variances = []
    predicted_means = []
    predicted_variances = []
    for n in range(len(measurements)):
        Z = measurements[n]
        X_updated, P_updated = update(h, sigma_v, Z, X_est_prior ,P_prior)
        X_predicted, P_predicted = predict(sigma_w, X_updated, P_updated, a = a, b = b, U = U)

        updated_means.append(X_updated)
        update_variances.append(P_updated)
        predicted_means.append(X_predicted)
        predicted_variances.append(P_predicted)
        
        X_est_prior = X_predicted
        P_prior = P_predicted
    return updated_means, update_variances, predicted_means, predicted_variances

