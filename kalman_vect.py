from numpy.random import uniform, seed
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib import pyplot as plt

def gauss_n_dimensional(x, y, Sigma, mu):
    X = np.vstack((x, y)).T
    mat_multi = np.dot((X-mu[None, ...]).dot(np.linalg.inv(Sigma)), (X-mu[None, ...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_countour(x, y, z, limits, color='r', label=''):
    # define grid.
    xi = np.linspace(limits[0], limits[1], 100)
    yi = np.linspace(limits[2], limits[3], 100)
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

def plot_2d_gauss(mu, Sigma, npts=1000, color='r', limits=[-1, 2, -1.5, 2.5], label=''):
    if Sigma.shape[0] == 1:
        infty = Sigma[0]*100000
        Sigma = np.array([[Sigma[0], 0], [0, infty]])
    x = uniform(limits[0], limits[1], npts)
    y = uniform(limits[2], limits[3], npts)
    z = gauss_n_dimensional(x, y, Sigma, mu)
    plot_countour(x, y, z, limits=limits, color=color, label=label)


def generate_sample(X_o, sigma_w, sigma_v, H, F, B, U, steps=10):
    HdotX = H.dot(X_o)
    measurements = [HdotX + np.random.multivariate_normal(np.zeros(len(HdotX)), sigma_v)]
    X_k = X_o
    for i in range(steps):
        X_k = F.dot(X_k) + B.dot(U) + np.random.multivariate_normal(np.zeros(len(X_k)), sigma_w).reshape(len(X_k), 1)
        HdotX = H.dot(X_k)
        Z = HdotX + np.random.multivariate_normal(np.zeros(len(HdotX)), sigma_v)
        measurements.append(Z)
    return np.array(measurements)

def predict(F, mu_x, sigma_x, Q):
    X_est = F.dot(mu_x)
    P = F.dot(sigma_x).dot(F.T) + Q
    return X_est, P

def update(H, R, Z, X_est_prior, P_est_prior):
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


def plot_prediction(F, mu_x, sigma_x, Q, y_offset=0.1, npts=1000, limit=1):
    mu_w = np.array([0, 0])
    plt.figure(figsize=(12, 6))
    plot_2d_gauss(mu_x, sigma_x, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='r', npts=npts)
    #plot_2d_gauss(mu_w, Q, limits = [-limit,3*limit, -limit+y_offset,limit+y_offset], color = 'y')

    mu_x, sigma_x = predict(F, mu_x, sigma_x, Q)
    plot_2d_gauss(mu_x, sigma_x, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='b', npts=npts)
    return mu_x, sigma_x

def plot_update(H, R, Z, X_est_prior, P_est_prior, y_offset=0.1, npts=1000, limit=1):
    R = np.array(R)
    Z = np.array(Z)
    H = np.array(H).reshape(R.shape[0], X_est_prior.shape[0])

    plt.figure(figsize=(12, 6))
    plot_2d_gauss(X_est_prior, P_est_prior, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='r', npts=npts)
    plot_2d_gauss(Z, R, limits=[-limit, 3*limit, -limit+y_offset, limit+y_offset], color='y')

    X_est, P = update(H, R, Z, X_est_prior, P_est_prior)
    plot_2d_gauss(X_est, P, limits=[-limit,3*limit, -limit+y_offset, limit+y_offset], color='b')
    return X_est, P
