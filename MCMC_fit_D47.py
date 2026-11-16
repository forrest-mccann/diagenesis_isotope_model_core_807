import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
from numba import njit, prange
from scipy.special import logsumexp

from diagenesis_model_D47 import DiagenesisModel_D47, DiagenesisModel_D47_expR


@njit(cache=True, parallel=True)
def _log_likelihood_D47_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top, time_bottom, data_bottom, var_bottom):
    '''
     calculate log likelihood for one theta, but a batch of boundary conditions
    '''
    n_draws = T_deep1_all.shape[0]
    lik_m = np.empty(n_draws)

    for i in prange(n_draws):
        model_top, model_bottom = DiagenesisModel_D47(theta, tspace1, tspace2, T_deep1_all[i], T_deep2_all[i], SST1_all, SST2_all, geotherm1_all[i], geotherm2_all[i])

        loglik_top = 0.0
        for j in range(len(time_top)):
            prediction = np.interp(time_top[j], tspace1, model_top)
            loglik_top += -0.5 * ( (data_top[j] - prediction) ** 2 / var_top[j] + np.log(2.0 * np.pi * var_top[j]) )

        loglik_bottom = 0.0
        for j in range(len(time_bottom)):
            prediction = np.interp(time_bottom[j], tspace2, model_bottom)
            loglik_bottom += -0.5 * ( (data_bottom[j] - prediction) ** 2 / var_bottom[j] + np.log(2.0 * np.pi * var_bottom[j]) )

        lik_m[i] = loglik_top + loglik_bottom

    return lik_m


def log_likelihood_D47_4data(theta, boundary_conditions_and_data, n_draws=5):
    '''
    calculate log likelihood for full sediment core, preps SST
    '''
    bcd = boundary_conditions_and_data
    rng = np.random.default_rng()

    obs_time = bcd.tspace_data.cpu().numpy()
    obs_D47 = bcd.D47.cpu().numpy()
    obs_err = bcd.D47_err.cpu().numpy()

    top_mask = obs_time <= bcd.t_switch
    bottom_mask = ~top_mask

    time_top = obs_time[top_mask]
    data_top = obs_D47[top_mask]
    var_top = obs_err[top_mask] ** 2 + theta[3] ** 2

    time_bottom = obs_time[bottom_mask]
    data_bottom = obs_D47[bottom_mask]
    var_bottom = obs_err[bottom_mask] ** 2 + theta[3] ** 2

    (tspace1, tspace2, T_deep1_all, T_deep2_all, depth1, depth2, SST_basis, SST_mean_norm, SSTH_data_mean, SSTH_data_std, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, indicies) = bcd.draw_all_BCs_batch(n_draws, rng)
    
    #calculate SST
    SST_guess_norm = SST_mean_norm + SST_basis@theta[4:]
    SST_guess = SST_guess_norm*SSTH_data_std + SSTH_data_mean
    SST1_all = SST_guess[:indicies[1]]
    SST2_all = SST_guess[indicies[1]-1:]

    lik_m = _log_likelihood_D47_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top, time_bottom, data_bottom, var_bottom)
    
    return logsumexp(lik_m) - np.log(n_draws)

@njit(cache=True, parallel=True)
def _log_likelihood_D47_4top_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top):
    '''
    calculate log likelihood for one theta, but a batch of boundary conditions
    '''
    n_draws = T_deep1_all.shape[0]
    lik_m = np.empty(n_draws)

    for i in prange(n_draws):
        model_top, model_bottom = DiagenesisModel_D47_expR(theta, tspace1, tspace2, T_deep1_all[i], T_deep2_all[i], SST1_all, SST2_all, geotherm1_all[i], geotherm2_all[i])

        loglik_top = 0.0
        for j in range(len(time_top)):
            prediction = np.interp(time_top[j], tspace1, model_top)
            loglik_top += -0.5 * ( (data_top[j] - prediction) ** 2 / var_top[j] + np.log(2.0 * np.pi * var_top[j]))

        lik_m[i] = loglik_top

    return lik_m


def log_likelihood_D47_4top_4data(theta, boundary_conditions_and_data, n_draws=5):
    '''
    calculate log likelihood for top sediment core, preps SST
    '''
    bcd = boundary_conditions_and_data
    rng = np.random.default_rng()

    obs_time = bcd.tspace_data.cpu().numpy()
    obs_d18O = bcd.D47.cpu().numpy()
    obs_err = bcd.D47_err.cpu().numpy()

    top_mask = obs_time <= bcd.t_switch
    bottom_mask = ~top_mask

    time_top = obs_time[top_mask]
    data_top = obs_d18O[top_mask]
    var_top = obs_err[top_mask] ** 2 + theta[3] ** 2

    (tspace1, tspace2, T_deep1_all, T_deep2_all, depth1, depth2, SST_basis, SST_mean_norm, SSTH_data_mean, SSTH_data_std, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, indicies) = bcd.draw_all_BCs_batch(n_draws, rng)
    
    #calculate SST
    SST_guess_norm = SST_mean_norm + SST_basis@theta[4:]
    SST_guess = SST_guess_norm*SSTH_data_std + SSTH_data_mean
    SST1_all = SST_guess[:indicies[1]]
    SST2_all = SST_guess[indicies[1]-1:]

    lik_m = _log_likelihood_D47_4top_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top)

    return logsumexp(lik_m) - np.log(n_draws)

def log_prior(theta):
    if (0 < theta[0] < 1) and (0 < theta[1] < 50) and (0 <= theta[2] < 1) and (0 <= theta[3] < 3):
        return -0.5*np.sum(theta[4:]**2)
    return -np.inf

def log_probability_D47_4data(theta, boundary_conditions_and_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_D47_4data(theta, boundary_conditions_and_data)

def log_probability_D47_4top_4data(theta, boundary_conditions_and_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_D47_4top_4data(theta, boundary_conditions_and_data)


## define neg likelihood probability distribution with interpolation for real data points
def neg_log_likelihood_D47_4data_mean(theta, boundary_conditions_and_data):
    bcd = boundary_conditions_and_data

    (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()

    model_top, model_bottom = DiagenesisModel_D47(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)

    obs_time = bcd.tspace_data.cpu().numpy()
    obs_d18O = bcd.D47.cpu().numpy()
    obs_var = bcd.D47_err.cpu().numpy() ** 2 + theta[3] ** 2

    top_mask = obs_time <= bcd.t_switch
    bottom_mask = ~top_mask

    pred_top = np.interp(obs_time[top_mask], tspace1, model_top)
    pred_bottom = np.interp(obs_time[bottom_mask], tspace2, model_bottom)

    loglik_top = -0.5 * np.sum( (obs_d18O[top_mask] - pred_top) ** 2 / obs_var[top_mask] + np.log(2.0 * np.pi * obs_var[top_mask]) )
    loglik_bottom = -0.5 * np.sum( (obs_d18O[bottom_mask] - pred_bottom) ** 2 / obs_var[bottom_mask] + np.log(2.0 * np.pi * obs_var[bottom_mask]) )

    return -(loglik_top + loglik_bottom)

## define neg likelihood probability distribution with alpha = 0 with interpolation for real data points
def neg_log_likelihood_D47_4top_4data_mean(theta, boundary_conditions_and_data):
    bcd = boundary_conditions_and_data
    theta = np.array([theta[0], theta[1], 0, theta[2]])

    (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()

    model_top, model_bottom = DiagenesisModel_D47_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)

    obs_time = bcd.tspace_data.cpu().numpy()
    obs_d18O = bcd.D47.cpu().numpy()
    obs_var = bcd.D47_err.cpu().numpy() ** 2 + theta[3] ** 2

    top_mask = obs_time <= bcd.t_switch

    pred_top = np.interp(obs_time[top_mask], tspace1, model_top)

    loglik_top = -0.5 * np.sum( (obs_d18O[top_mask] - pred_top) ** 2 / obs_var[top_mask] + np.log(2.0 * np.pi * obs_var[top_mask]) )

    return -(loglik_top)
