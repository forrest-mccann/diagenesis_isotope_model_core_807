import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from functools import partial
from scipy.optimize import minimize
from numba import njit, prange
from scipy.special import logsumexp

from diagenesis_model_d18O import DiagenesisModel_d18O, DiagenesisModel_d18O_expR
from plotter import plot_d18O, plot_R, plot_d18O_expR, plot_R_expR, plot_D47, plot_D47_expR


@njit(cache=True, parallel=True)
def _log_likelihood_d18O_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, seawater1_all, seawater2_all,geotherm1_all, geotherm2_all, time_top, data_top, var_top, time_bottom, data_bottom, var_bottom):
    '''
    calculated log likelihood for one theta, but a batch of boundary conditions
    '''
    n_draws = T_deep1_all.shape[0]
    lik_m = np.empty(n_draws)

    for i in prange(n_draws):
        model_top, model_bottom = DiagenesisModel_d18O(theta, tspace1, tspace2, T_deep1_all[i], T_deep2_all[i], SST1_all, SST2_all, seawater1_all, seawater2_all, geotherm1_all[i], geotherm2_all[i])

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

def log_likelihood_d18O_4data(theta, boundary_conditions_and_data, n_draws=30):
    '''
    calculate log likelihood for full sediment core, preps SST
    '''
    bcd = boundary_conditions_and_data
    rng = np.random.default_rng()

    obs_time = bcd.tspace_data_d18O.cpu().numpy()
    obs_d18O = bcd.d18O.cpu().numpy()
    obs_err = bcd.d18O_err.cpu().numpy()

    top_mask = obs_time <= bcd.t_switch
    bottom_mask = ~top_mask

    time_top = obs_time[top_mask]
    data_top = obs_d18O[top_mask]
    var_top = obs_err[top_mask] ** 2 + theta[3] ** 2

    time_bottom = obs_time[bottom_mask]
    data_bottom = obs_d18O[bottom_mask]
    var_bottom = obs_err[bottom_mask] ** 2 + theta[3] ** 2


    (tspace1, tspace2, T_deep1_all, T_deep2_all, depth1, depth2, SST_basis, SST_mean_norm, SSTH_data_mean, SSTH_data_std, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, indicies) = bcd.draw_all_BCs_batch(n_draws, rng)

    #calculate SST
    SST_guess_norm = SST_mean_norm + SST_basis@theta[4:]
    SST_guess = SST_guess_norm*SSTH_data_std + SSTH_data_mean
    SST1_all = SST_guess[:indicies[1]]
    SST2_all = SST_guess[indicies[1]-1:]

    
    lik_m = _log_likelihood_d18O_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top, time_bottom, data_bottom, var_bottom)
   
    return logsumexp(lik_m) - np.log(n_draws)

@njit(cache=True, parallel=True)
def _log_likelihood_d18O_4top_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, seawater1_all, seawater2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top):
    '''
    calculated log likelihood for one theta, but a batch of boundary conditions
    '''
    n_draws = T_deep1_all.shape[0]
    lik_m = np.empty(n_draws)

    for i in prange(n_draws):
        model_top, model_bottom = DiagenesisModel_d18O_expR(theta, tspace1, tspace2, T_deep1_all[i], T_deep2_all[i], SST1_all, SST2_all, seawater1_all, seawater2_all, geotherm1_all[i], geotherm2_all[i])

        loglik_top = 0.0
        for j in range(len(time_top)):
            prediction = np.interp(time_top[j], tspace1, model_top)
            loglik_top += -0.5 * ( (data_top[j] - prediction) ** 2 / var_top[j] + np.log(2.0 * np.pi * var_top[j]))

        lik_m[i] = loglik_top

    return lik_m

def log_likelihood_d18O_4top_4data(theta, boundary_conditions_and_data, n_draws=30):
    '''
    calculate log likelihood for top sediment core, preps SST
    '''
    bcd = boundary_conditions_and_data
    rng = np.random.default_rng()

    obs_time = bcd.tspace_data_d18O.cpu().numpy()
    obs_d18O = bcd.d18O.cpu().numpy()
    obs_err = bcd.d18O_err.cpu().numpy()

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
    
    lik_m = _log_likelihood_d18O_4top_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_all, SST2_all, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top)

    return logsumexp(lik_m) - np.log(n_draws)

def log_prior(theta):
    if (0 < theta[0] < 1) and (0 < theta[1] < 50) and (0 <= theta[2] < 1) and (0 <= theta[3] < 3):
        return -0.5*np.sum(theta[4:]**2)
    return -np.inf

def log_probability_d18O_4data(theta, boundary_conditions_and_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_d18O_4data(theta, boundary_conditions_and_data)

def log_probability_d18O_4top_4data(theta, boundary_conditions_and_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_d18O_4top_4data(theta, boundary_conditions_and_data)

## define neg likelihood probability distribution with interpolation for real data points
def neg_log_likelihood_d18O_4data_mean(theta, boundary_conditions_and_data):
    bcd = boundary_conditions_and_data

    (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()

    model_top, model_bottom = DiagenesisModel_d18O(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2)

    obs_time = bcd.tspace_data_d18O.cpu().numpy()
    obs_d18O = bcd.d18O.cpu().numpy()
    obs_var = bcd.d18O_err.cpu().numpy() ** 2 + theta[3] ** 2

    top_mask = obs_time <= bcd.t_switch
    bottom_mask = ~top_mask

    pred_top = np.interp(obs_time[top_mask], tspace1, model_top)
    pred_bottom = np.interp(obs_time[bottom_mask], tspace2, model_bottom)

    loglik_top = -0.5 * np.sum( (obs_d18O[top_mask] - pred_top) ** 2 / obs_var[top_mask] + np.log(2.0 * np.pi * obs_var[top_mask]) )
    loglik_bottom = -0.5 * np.sum( (obs_d18O[bottom_mask] - pred_bottom) ** 2 / obs_var[bottom_mask] + np.log(2.0 * np.pi * obs_var[bottom_mask]) )

    return -(loglik_top + loglik_bottom)

## define neg likelihood probability distribution with alpha = 0 with interpolation for real data points
def neg_log_likelihood_d18O_4top_4data_mean(theta, boundary_conditions_and_data):
    bcd = boundary_conditions_and_data
    theta = np.array([theta[0], theta[1], 0, theta[2]])

    (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()

    model_top, model_bottom = DiagenesisModel_d18O_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2)

    obs_time = bcd.tspace_data_d18O.cpu().numpy()
    obs_d18O = bcd.d18O.cpu().numpy()
    obs_var = bcd.d18O_err.cpu().numpy() ** 2 + theta[3] ** 2

    top_mask = obs_time <= bcd.t_switch

    pred_top = np.interp(obs_time[top_mask], tspace1, model_top)

    loglik_top = -0.5 * np.sum( (obs_d18O[top_mask] - pred_top) ** 2 / obs_var[top_mask] + np.log(2.0 * np.pi * obs_var[top_mask]) )

    return -(loglik_top)

# set up MCMC class
class diag_MCMC:
    '''
    Works for both d18O and D47. Takes in necessary values MCMC run (walkers, nsteps, burnin), 
    the d18O/D47 top/full probability function (log_prob_fn), the boundary conditions and data, 
    the maximum likelihood estimate (MLE_estimate), path for saving outputs, 
    and the backend for storing the MCMC run in case it needs to be run for longer.
    '''
    def __init__(self, walkers, nsteps, burnin, log_prob_fn, boundary_conditions_and_data, MLE_estimate, path = './', use_backend = None):

        log_prob = partial(log_prob_fn, boundary_conditions_and_data = boundary_conditions_and_data)

            
        n = len(MLE_estimate)
        # pos = np.zeros((walkers, n))
        pos = np.random.rand(walkers, n)

        pos[:, 0] = MLE_estimate[0] + 1e-4*np.random.uniform(-1, 1, walkers) 
        pos[:, 1] = MLE_estimate[1] + 1e-4*np.random.uniform(-50, 50, walkers) 
        pos[:, 2] = MLE_estimate[2] + 1e-4*np.random.uniform(-1, 1, walkers)
        pos[:, 3] = MLE_estimate[3] + 1e-4*np.random.uniform(-1, 1, walkers)

        nwalkers, ndim = pos.shape

        backend = emcee.backends.HDFBackend(path + "chain.h5")

        if use_backend is not None:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,  moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)], backend=backend)
            sampler.run_mcmc(None, nsteps, progress=True)

        if use_backend is None:
            backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)], backend=backend)
            sampler.run_mcmc(pos, nsteps, progress=True)

        self.sampler = sampler
        samples = backend.get_chain()
        flat_samples = backend.get_chain(discard=burnin, flat=True)
        self.flat_samples = flat_samples

        theta = np.median(flat_samples, axis=0)

        self.params = theta
        self.beta = np.percentile(flat_samples[:, 0], [16, 50, 84])
        self.gamma = np.percentile(flat_samples[:, 1], [16, 50, 84])
        self.alpha = np.percentile(flat_samples[:, 2], [16, 50, 84])
        self.model_sigma = np.percentile(flat_samples[:, 3], [16, 50, 84])
        
        #plot results
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        # labels = ['b', 'g', 'a', 'model_s']
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            # ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        fig.savefig(path + "chain_plot.png", dpi=300, bbox_inches='tight')

        try:
            self.tau = sampler.get_autocorr_time()
            print("Autocorrelation time:", self.tau)

        except emcee.autocorr.AutocorrError as err:
            self.tau = None
            print("Chain is too short for a reliable autocorrelation-time estimate yet. ", f"{err}")

        fig = corner.corner(flat_samples)#, labels=labels)
        fig.savefig(path + "corner_plot.png", dpi=300, bbox_inches='tight')

init_guess = np.array([1.05283099e-01,  6.27005043e+00,  3.77516840e-03,  2.57891941e-01])

def MLE_top_core(NLL_function, boundary_conditions_and_data, plot = False, initial_guess = init_guess): 
    '''
    Performs a maximum likelihood estimation (MLE) optimization for the top part of the sediment core data
    '''
       
    init_params = np.array([initial_guess[0], initial_guess[1], initial_guess[3]])
    
    NLL_top = partial(NLL_function, boundary_conditions_and_data = boundary_conditions_and_data)
        
        
    result = minimize(NLL_top, init_params, method="L-BFGS-B",
        bounds=[
            (1e-8, 1.0 - 1e-8),  # beta
            (1e-8, 50.0 - 1e-8), # gamma
            (1e-8, 3.0 - 1e-8), # model_s
        ], options={'eps': np.array([1e-4, 1e-1, 1e-2])} )

    theta = np.array([result.x[0], result.x[1], 0, result.x[2]])

    if plot == True:
        plot_d18O_expR(theta = theta, boundary_conditions_and_data = boundary_conditions_and_data)
        plot_D47_expR(theta = theta, boundary_conditions_and_data = boundary_conditions_and_data)
        plot_R_expR(theta = theta, boundary_conditions_and_data = boundary_conditions_and_data)

        integrated_R_top = theta[0]*theta[1]*(1-np.exp(-1*boundary_conditions_and_data.t_switch/theta[1]))
        print("Integrated R top: ", integrated_R_top)

    return theta

def MLE_full_core(NLL_function, boundary_conditions_and_data, plot = False, initial_guess = init_guess):  
    '''
    Performs a maximum likelihood estimation (MLE) optimization for the full sediment core data
    '''  
    
    init_params = initial_guess
    
    NLL_full = partial(NLL_function, boundary_conditions_and_data = boundary_conditions_and_data)
        
        
    result = minimize(NLL_full, init_params, method="L-BFGS-B",
        bounds=[
            (1e-8, 1.0 - 1e-8),  # beta
            (1e-8, 50.0 - 1e-8), # gamma
            (0.0, 1.0 - 1e-8),   # alpha
            (0.0, 3.0 - 1e-8),   # model_s
        ], options={'eps': np.array([1e-4, 1e-1, 1e-4, 1e-2])} )

    theta = result.x

    if plot == True:
        plot_d18O(theta = theta, boundary_conditions_and_data = boundary_conditions_and_data)
        plot_D47(theta = theta, boundary_conditions_and_data = boundary_conditions_and_data)
        plot_R(theta = theta, boundary_conditions_and_data = boundary_conditions_and_data)

        integrated_R_top = theta[0]*theta[1]*(1-np.exp(-1*boundary_conditions_and_data.t_switch/theta[1]))
        print("Integrated R top: ", integrated_R_top)

    return theta