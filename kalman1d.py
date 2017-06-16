import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import *

## Auxiliary functions ##
def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.)/2)/(np.sqrt(2.*np.pi)*sig)

## Ploting functions ##
def plot_gaussian(mu=0, sig=1, points=200, N=2, marker=None, label=None, x=None, color=None,
                  x_label='', y_label='', ax=None, figsize=(20, 10), center_label_tick=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=figsize)
    if x is None:
        x = np.linspace(mu-N*sig, mu+N*sig, points)
    y = gaussian(x, mu, sig)
    ax.plot(x, y, marker=marker, label=label, color=color)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if center_label_tick is not None:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[int((len(labels)-1)/2)] = center_label_tick
        ax.set_xticklabels(labels)

    return x, y

def plot_filter_densities(ax, X_est_prior, P_prior, X_updated, P_updated, X_predicted, P_predicted,
                          Z=None, actual_position=None, points=200, x_limits=None, N_stds=2):
    if x_limits is None:
        X_array = np.array([X_est_prior, X_updated, X_predicted])
        p_array = np.array([P_prior, P_updated, P_predicted])
        X_min_index = np.argmin(X_array)
        X_max_index = np.argmax(X_array)

        x = np.linspace(X_array[X_min_index]-N_stds*(p_array[X_min_index]**0.5),
                        X_array[X_max_index]+N_stds*(p_array[X_max_index]**0.5), points)
    else:
        x = np.linspace(x_limits[0], x_limits[1], points)

    label_data = r'$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_est_prior, P_prior)
    plot_gaussian(mu=X_est_prior, sig=P_prior, points=points, N=2, x=x,
                  label='(Prior) '+label_data, color='k', ax=ax)

    label_data = r'$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_updated, P_updated)
    plot_gaussian(mu=X_updated, sig=P_updated, points=points, N=2, x=x,
                  label='(Update) '+label_data, color='b', ax=ax)

    label_data = r'$\mu=%0.2f$  -  $\sigma^2=%0.2f$'%(X_predicted, P_predicted)
    plot_gaussian(mu=X_predicted, sig=P_predicted, points=points, N=2, x=x,
                  label='(Predict) '+label_data, color='y', ax=ax)

    if Z is not None:
        ax.scatter(Z, 0, s=100, color="r", alpha=0.5, label='measurement. Z=%.2f'%Z)

    if actual_position is not None:
        ax.scatter(actual_position, 0, s=100, color="g", alpha=0.5,
                   label='actual position=%.2f'%actual_position)

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
    updated_means, update_variances, predicted_means, predicted_variances, kalman_gains = kalman_filter(measurements_limited, 
                                                                                      X_o, P_0, 
                                                                                      sigma_v, 
                                                                                      sigma_w, h = 1, a = 1, 
                                                                                      b = 1, U = U)
    
    X_est_prior = predicted_means[-2]
    P_prior = predicted_variances[-2]
    X_updated = updated_means[-1]
    P_updated = update_variances[-1]
    X_predicted = predicted_means[-1]
    P_predicted = predicted_variances[-1]
    actual_position = None
    if real_positions is not None:
        actual_position = real_positions_limited[-1]

    f = plt.figure(figsize=(20, 14))
    ax = plt.subplot(3, 1, 1)

    plot_filter_densities(ax, X_est_prior, P_prior, X_updated, P_updated,X_predicted, P_predicted, Z=measurements_limited[-1], 
                            actual_position = actual_position, points = points, N_stds = N_stds, 
                            x_limits = x_limits)
    
    ax.set_xlabel('POSITION')

    ax = plt.subplot(3, 2, 3)
    plot_kalman_filter_results(updated_means, predicted_means, measurements_limited, real_positions_limited)
    ax.set_xlabel('iteration number')

    ax = plt.subplot(3, 2, 4)
    plt.plot(update_variances, label=r'Update variance $P_{n|n}$')
    plt.plot(predicted_variances, label=r'Prediction variance $P_{n|n-1}$')
    plt.legend()
    ax.set_xlabel('iteration number')

    ax = plt.subplot(3, 1, 3)
    plt.plot(kalman_gains, label='Kalman Gain')
    ax.set_xlabel('iteration number')
    

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


## Filter functions ##
def update(h, sigma_v, Z, X_est_prior, P_prior):
    P = P_prior*sigma_v/(P_prior*(h**2) + sigma_v)
    X_est = P*(h*Z/sigma_v + X_est_prior/P_prior)
    return X_est, P

def predict(sigma_w, X_updated, P_updated, a=1, b=1, U=1):
    X_predicted = a*X_updated + b*U
    P_predicted = (a**2)*sigma_w + P_updated
    return X_predicted, P_predicted

def update_with_kalman_gain(h, sigma_v, Z, X_est_prior, P_prior):
    K = P_prior*h/(P_prior*h**2+sigma_v)
    P = P_prior*(1-h*K)
    X_est = X_est_prior + K*(Z-h*X_est_prior)
    return X_est, P, K

def kalman_filter(measurements, X_est_prior, P_prior, sigma_v, sigma_w, h=1, a=1, b=1, U=1):
    updated_means = []
    update_variances = []
    predicted_means = []
    predicted_variances = []
    kalman_gains = []
    for n, Z in enumerate(measurements):

        X_updated, P_updated, K = update_with_kalman_gain(h, sigma_v, Z, X_est_prior, P_prior)
        X_predicted, P_predicted = predict(sigma_w, X_updated, P_updated, a=a, b=b, U=U)

        updated_means.append(X_updated)
        update_variances.append(P_updated)
        predicted_means.append(X_predicted)
        predicted_variances.append(P_predicted)
        kalman_gains.append(K)

        X_est_prior = X_predicted
        P_prior = P_predicted
    return updated_means, update_variances, predicted_means, predicted_variances, kalman_gains

def generate_sample(X_o=0, sigma_w=0.1, sigma_v=0.1, h=1, a=1, b=1, U=1, steps=10):
    real_positions = [X_o]
    measurements = [h*X_o + np.random.normal(0,sigma_v)]
    X_k = X_o
    for i in range(steps):
        X_k = a*X_k + b*U + np.random.normal(0,sigma_w)
        real_positions.append(X_k)
        Z = h*X_k + np.random.normal(0,sigma_v)
        measurements.append(Z)
    return measurements, real_positions


class Kalman1D:
    def __init__(self, sigma_v=10, sigma_w=0.1, X_0=0, P_0=50, h=1, a=1, b=1, U=1):
        # Observation variance
        self.sigma_v = sigma_v
        # Process Noise
        self.sigma_w = sigma_w
        # Priors
        self.X_0 = X_0
        self.P_0 = P_0
        self.h = h
        self.a = a
        self.b = b
        self.U = U
        self.measurements = None
        self.ground_truths = None
        self.updated_Xs = None
        self.predicted_Xs = None
        self.updated_Ps = None
        self.predicted_Ps = None
        self.kalman_gains = None

    def __repr__(self):
        return ('Model parameters:\nObservation Noise Variance: %s\nProcess Noise Variance: %s\nInitial guess: %s\nInitial uncertainty: %s\nh=%s, a=%s, b=%s, U=%s' 
                % (repr(self.sigma_v), repr(self.sigma_w), repr(self.X_0), repr(self.P_0), repr(self.h), repr(self.a), repr(self.b), repr(self.U)))

    def set_measurements(measurements, ground_truths=None):
        self.measurements = measurements
        self.ground_truths = ground_truths

    def filter(self, measurements = None, ground_truths = None):
        if (measurements is None) and (self.measurements is None):
            print('No measurements available to filter. Pass parameter measurements, generate_model_samples, or set_measuremnets')
            return
        elif measurements is not None:
            self.measurements = measurements
            self.ground_truths = ground_truths
        
        
        self.updated_Xs, self.updated_Ps, self.predicted_Xs, self.predicted_Ps, self.kalman_gains =\
                        kalman_filter(self.measurements, 
                        self.X_0, self.P_0, self.sigma_v, self.sigma_w, h=self.h, a=self.a, b=self.b, U=self.U)


    def plot_kalman_filter_steps(self, number_of_graphs=12, points=200):
        measurements = self.measurements[:number_of_graphs]

        rows = int(np.ceil(len(measurements)/3))
        f, ax = plt.subplots(rows, 3, sharey=True, sharex=True, figsize=(20, 10))
        ax = ax.flatten()
        x_min = min(measurements)
        x_max = max(measurements)
        actual_position = None
        X_est_prior = self.X_0
        P_prior = self.P_0
        for n in range(len(measurements)):
            Z = measurements[n]
            if self.ground_truths is not None:
                actual_position = self.ground_truths[n]

            X_updated = self.updated_Xs[n]
            P_updated = self.updated_Ps[n]

            X_predicted = self.predicted_Xs[n]
            P_predicted = self.predicted_Ps[n]

            plot_filter_densities(ax[n], X_est_prior, P_prior, X_updated, P_updated, X_predicted,
                                  P_predicted, Z=Z, actual_position=actual_position,
                                  points=points, x_limits=[x_min, x_max])

            X_est_prior = X_predicted
            P_prior = P_predicted
        plt.show()

    def generate_model_samples(self, iterations = 20):
        measurements, ground_truths = generate_sample(X_o=self.X_0, sigma_w=self.sigma_w, 
                                                        sigma_v=self.sigma_v,
                                                        h=self.h, a=self.a, b=self.b, U=self.U,
                                                        steps=iterations)
        self.measurements = measurements
        self.ground_truths = ground_truths
        return measurements, ground_truths

    def plot_interactive_kalman_filter(self, max_number_of_steps=20 , initial_slider_pos=5, x_limits=None, N_stds=3):
        plot_interactive_kalman_filter(self.measurements, self.X_0, self.P_0, 
                                       self.sigma_v, self.sigma_w, max_number_of_steps, 
                                       h = self.h, a = self.a, b = self.b, U = self.U,
                                       N_stds=N_stds, real_positions=self.ground_truths , 
                                       x_limits = x_limits, 
                                       initial_slider_pos=initial_slider_pos)