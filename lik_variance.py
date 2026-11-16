#import packages and functions
import numpy as np
from scipy.special import logsumexp

from MCMC_fit_d18O import _log_likelihood_d18O_batch, _log_likelihood_d18O_4top_batch
from MCMC_fit_D47 import _log_likelihood_D47_4top_batch, _log_likelihood_D47_batch


def diagnose_lik_variance(theta, boundary_conditions_and_data, full_core, data_type, n_draws=100, n_repeats=30):
    """
    function to calculate Var(log p_hat(D|theta)) at a fixed theta
    """
    bcd = boundary_conditions_and_data
    rng = np.random.default_rng()

    if data_type == 'd18O':
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

    if data_type == 'D47':
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

    log_phat_estimates = [] 
    all_lik_m_stds = [] 
    all_lik_m_ess = []

    (tspace1_mean, tspace2_mean, T_deep1_mean, T_deep2_mean, depth1_mean, depth2_mean, SST1_mean, SST2_mean, seawater_d18O1_mean, seawater_d18O2_mean, geotherm1_mean, geotherm2_mean) = bcd.mean_all_BCs()

    for r in range(n_repeats):
        (tspace1, tspace2, T_deep1_all, T_deep2_all, depth1, depth2, SST_basis, SST_mean_norm, SSTH_data_mean, SSTH_data_std, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, indicies) = bcd.draw_all_BCs_batch(n_draws, rng)

        if full_core == True:
            if data_type == 'd18O':
                lik_m = _log_likelihood_d18O_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_mean, SST2_mean, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top, time_bottom, data_bottom, var_bottom) 
            if data_type == 'D47':
                lik_m = _log_likelihood_D47_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_mean, SST2_mean, geotherm1_all, geotherm2_all, time_top, data_top, var_top, time_bottom, data_bottom, var_bottom)

        if full_core == False:
            if data_type == 'd18O':
                lik_m = _log_likelihood_d18O_4top_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_mean, SST2_mean, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, time_top, data_top, var_top)
            if data_type == 'D47':
                lik_m = _log_likelihood_D47_4top_batch(theta, tspace1, tspace2, T_deep1_all, T_deep2_all, SST1_mean, SST2_mean, geotherm1_all, geotherm2_all, time_top, data_top, var_top)

        log_phat = logsumexp(lik_m) - np.log(n_draws)
        log_phat_estimates.append(log_phat)
        all_lik_m_stds.append(np.std(lik_m))

        # effective sample size of the importance weights within this one batch
        w = np.exp(lik_m - lik_m.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        all_lik_m_ess.append(ess)

    log_phat_estimates = np.array(log_phat_estimates)

    print('Var(log p_hat(D|theta)): ', log_phat_estimates.var())
