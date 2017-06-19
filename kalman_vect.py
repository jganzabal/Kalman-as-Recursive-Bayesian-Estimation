from numpy.random import uniform, seed
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib import pyplot as plt

def gauss_n_dimensional(x, y, Sigma, mu):
    X = np.vstack((x, y)).T
    X_centered = X-np.array(mu).flatten()
    mat_multi = np.dot(X_centered.dot(Sigma.I), X_centered.T)
    
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_countour(x, y, z, limits, color='r', label=''):
    # define grid.
    xi = np.linspace(limits[0], limits[1], 400)
    yi = np.linspace(limits[2], limits[3], 400)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    levels = [0.85, 0.9, 0.95]
    n_levels = len(levels)
    #n_levels = 10
    # contour the gridded data, plotting dots at the randomly spaced data points.
    #CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    #CS = plt.contourf(xi,yi,zi,n_levels,cmap=cm.Greys_r, levels=levels)
    #CS = plt.contourf(xi,yi,zi,n_levels,cmap=cm.Greys_r)
    plt.contour(xi, yi, zi, n_levels, linewidths=0.5, colors=color, levels=levels, label=label)
    #plt.colorbar() # draw colorbar
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)

def plot_2d_gauss(mu, Sigma, x_npts=1000, y_npts=1000, color='r', limits=[-1, 2, -1.5, 2.5], label=''):
    if Sigma.shape[0] == 1:
        sigma_0 = Sigma[0,0]
        infty = sigma_0*100000
        Sigma = np.matrix([[sigma_0, 0], [0, infty]])

    x = uniform(limits[0], limits[1], x_npts)
    y = uniform(limits[2], limits[3], y_npts)
    z = gauss_n_dimensional(x, y, Sigma, mu)
    plot_countour(x, y, z, limits=limits, color=color, label=label)



def plot_prediction(F, mu_x, sigma_x, Q, y_offset=0.1, npts=1000, limit=1):
    mu_w = np.array([0, 0])
    plt.figure(figsize=(12, 6))
    plot_2d_gauss(mu_x, sigma_x, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='r', npts=npts)
    #plot_2d_gauss(mu_w, Q, limits = [-limit,3*limit, -limit+y_offset,limit+y_offset], color = 'y')

    mu_x, sigma_x = predict(F, mu_x, sigma_x, Q)
    plot_2d_gauss(mu_x, sigma_x, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='b', npts=npts)
    return mu_x, sigma_x

def update(H, R, Z, X_est_prior ,P_est_prior):

    A_1 = H.T.dot(R.I).dot(H)

    P = np.linalg.inv(A_1 + P_est_prior.I)

    X = P.dot(H.T.dot(R.I).dot(Z) + P_est_prior.I.dot(X_est_prior))
    return X, P

def plot_update(H, R, Z, X_est_prior, P_est_prior, y_offset=0.1, x_npts=2000, y_npts=2000, limit=1):
    plt.figure(figsize=(12, 6))
    plot_2d_gauss(X_est_prior, P_est_prior, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='r', 
                  x_npts=x_npts, y_npts=y_npts)
    plot_2d_gauss(Z, R, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='y',
                  x_npts=x_npts, y_npts=y_npts)

    X_est, P = update(H, R, Z, X_est_prior, P_est_prior)
    plot_2d_gauss(X_est, P, limits=[-limit,3*limit, -limit+y_offset, limit+y_offset], color='b',
                  x_npts=x_npts, y_npts=y_npts)
    return X_est, P


def predict(F, mu_x, sigma_x, Q):
    X_est = F.dot(mu_x)
    P = F.dot(sigma_x).dot(F.T) + Q
    return X_est, P

def update_(H, R, Z, X_est_prior, P_est_prior):
    R = np.array(R)
    Z = np.array(Z)
    H = np.array(H).reshape(R.shape[0], X_est_prior.shape[0])

    if R.shape[0] == 1:
        A_1 = H.T.dot(H)/R[0]
    else:
        A_1 = H.T.dot(np.linalg.inv(R)).dot(H)

    P_est_prior_inv = np.linalg.inv(P_est_prior)

    P_est = np.linalg.inv(A_1 + P_est_prior_inv)

    if R.shape[0] == 1:
        X_est = P_est.dot(H.T.dot(Z)/R[0] + P_est_prior_inv.dot(X_est_prior))
    else:
        X_est = P_est.dot(H.T.dot(np.linalg.inv(R)).dot(Z) + P_est_prior_inv.dot(X_est_prior))

    return X_est, P_est

def kalman_filter(measurements, X_est_prior, P_prior, H, R, F, Q):
    updated_means = []
    update_covariances = []
    predicted_means = []
    predicted_covariances = []
    kalman_gains = []

    for n, Z in enumerate(measurements):
        
        X_updated, P_updated = update(H, R, Z, X_est_prior, P_est_prior)
        X_predicted, P_predicted = predict(F, mu_x, sigma_x, Q)

        updated_means.append(X_updated)
        update_covariances.append(P_updated)
        predicted_means.append(X_predicted)
        predicted_covariances.append(P_predicted)
        #kalman_gains.append(K)

        X_est_prior = X_predicted
        P_prior = P_predicted
    return updated_means, update_covariances, predicted_means, predicted_covariances #, kalman_gains

def generate_sample(X_0, Q, R, H, F, steps=10):
    HdotX = H.dot(X_0)
    measurements = [HdotX + np.random.multivariate_normal(np.zeros(R.shape[0]), R).reshape(-1,1)]
    ground_truths = [X_0]
    X_k = X_0
    for i in range(steps):
        X_k = F.dot(X_k) +  np.random.multivariate_normal(np.zeros(len(X_k)), Q).reshape(len(X_k), 1)
        ground_truths.append(X_k)
        HdotX = H.dot(X_k)
        Z = HdotX + np.random.multivariate_normal(np.zeros(R.shape[0]), R).reshape(-1,1)
        measurements.append(Z)
    return measurements, ground_truths


class Kalman:
    def __init__(self, X_0, P_0, H, R, F, Q):
        self.X_0 = np.matrix(X_0)
        self.P_0 = np.matrix(P_0)
        self.H = np.matrix(H)
        self.R = np.matrix(R)
        self.F = np.matrix(F)
        self.Q = np.matrix(Q)
        self.measurements = None
        self.ground_truths = None
        self.updated_Xs = None
        self.predicted_Xs = None
        self.updated_Ps = None
        self.predicted_Ps = None
        self.kalman_gains = None

    def __repr__(self):
        return ('Model parameters:\nObservation Covariance: %s\nProcess Covariance: %s\nInitial guess: %s\nInitial uncertainty: %s\nH=%s\nF=%s' 
                % (repr(self.R), repr(self.Q), repr(self.X_0), repr(self.P_0), repr(self.H), repr(self.F)))

    def generate_model_sample(self, X_0, iterations=10):
        X_0 = np.matrix(X_0)
        self.measurements, self.ground_truths = generate_sample(X_0, self.Q, self.R, self.H,
                                                                self.F, steps=iterations)
        return self.measurements, self.ground_truths

    def update(self, Z, X_est_prior, P_est_prior):
        Z = np.matrix(Z)
        X_est_prior = np.matrix(X_est_prior)
        P_est_prior = np.matrix(P_est_prior)

        H = self.H
        R = self.R
        A_1 = H.T.dot(R.I).dot(H)

        P = np.linalg.inv(A_1 + P_est_prior.I)

        X = P.dot(H.T.dot(R.I).dot(Z) + P_est_prior.I.dot(X_est_prior))

        return X, P

    def update_with_kalman_gain(self, Z, X_est_prior, P_est_prior):
        Z = np.matrix(Z)
        X_est_prior = np.matrix(X_est_prior)
        P_est_prior = np.matrix(P_est_prior)

        H = self.H
        R = self.R

        S = R + H.dot(P_est_prior).dot(H.T)

        K = P_est_prior.dot(H.T).dot(S.I)

        X = X_est_prior + K.dot(Z-H.dot(X_est_prior))

        P = (np.identity(P_est_prior.shape[0]) - K.dot(H)).dot(P_est_prior)

        return X, P, K

    def predict(self, X_est, P_est):
        X_est = np.matrix(X_est)
        P_est = np.matrix(P_est)

        F = self.F
        Q = self.Q

        X = F.dot(X_est)
        P = F.dot(P_est).dot(F.T) + Q
        return X, P

    def filter(self, measurements = None, X_est_prior=None, P_prior=None):
        if measurements is None:
            measurements = self.measurements
        if measurements is None:
            print('No measurements!')
            return

        if X_est_prior is None:
            X_est_prior = self.X_0
        else:
            X_est_prior = np.matrix(X_est_prior)
        if P_prior is None:
            P_prior = self.P_0
        else:
            P_prior = np.matrix(P_prior)

        H = self.H
        R = self.R
        F = self.F
        Q = self.Q

        updated_means = []
        update_covariances = []
        predicted_means = []
        predicted_covariances = []
        kalman_gains = []

        for n, Z in enumerate(measurements):
            X_updated, P_updated, K = self.update_with_kalman_gain(Z, X_est_prior, P_prior)
            X_predicted, P_predicted = self.predict(X_updated, P_updated)

            updated_means.append(X_updated)
            update_covariances.append(P_updated)
            predicted_means.append(X_predicted)
            predicted_covariances.append(P_predicted)
            kalman_gains.append(K)

            X_est_prior = X_predicted
            P_prior = P_predicted
        return updated_means, update_covariances, predicted_means, predicted_covariances, kalman_gains 