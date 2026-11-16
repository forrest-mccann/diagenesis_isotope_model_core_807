from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gpytorch, torch

from diagenesis_model_d18O import DiagenesisModel_d18O, DiagenesisModel_d18O_expR
from diagenesis_model_D47 import DiagenesisModel_D47, DiagenesisModel_D47_expR
from data_and_boundary_conditions import age2depth, depth2age

# set up requisite functions
def recryst_func(theta, time, t_switch):
    rate = np.empty_like(time, dtype=float)
    top = time < t_switch
    rate[top] = theta[0] * np.exp(-time[top] / theta[1])
    rate[~top] = theta[2]
    return rate

def recryst_func_top(theta, time):
    return theta[0] * np.exp(-time / theta[1])

# set up plotting class
class percentiles:
    """
    sample from posterior distributions of theta and all boundary conditions. Has function for generating percentiles and making figures. 
    """
    def __init__(self, flat_samples, boundary_conditions_and_data, fit_to = None, n_curves=500, full_core=True, seed=None, path="./", make_plots=True):
        bcd = boundary_conditions_and_data
        
        flat_samples = np.asarray(flat_samples, dtype=float)
        rng = np.random.default_rng(seed)
        n_curves = min(int(n_curves), len(flat_samples))

        sample_indices = rng.choice(len(flat_samples), size=n_curves, replace=False)
        theta_curves = flat_samples[sample_indices]

        curves_d18O = []
        curves_D47 = []
        curves_r = []
        curves_SST = []
        time = None

        (tspace1, tspace2, T_deep1_all, T_deep2_all, depth1, depth2, SST_basis, SST_mean_norm, SSTH_data_mean, SSTH_data_std, seawater_d18O1_all, seawater_d18O2_all, geotherm1_all, geotherm2_all, indicies) = bcd.draw_all_BCs_batch(n_curves, rng)
        i = 0
        for theta in theta_curves:
            T_deep1 = T_deep1_all[i]
            T_deep2 = T_deep2_all[1]
            seawater_d18O1 = seawater_d18O1_all
            seawater_d18O2 = seawater_d18O2_all
            geotherm1 = geotherm1_all[i]
            geotherm2 = geotherm2_all[i]

            SST_guess_norm = SST_mean_norm + SST_basis@theta[4:]
            SST_guess = SST_guess_norm*SSTH_data_std + SSTH_data_mean
            SST1 = SST_guess[:indicies[1]]
            SST2 = SST_guess[indicies[1]-1:]

            curves_SST.append(SST_guess)

            if full_core:
                model_top_d18O, model_bottom_d18O = DiagenesisModel_d18O(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2)
                model_top_D47, model_bottom_D47 = DiagenesisModel_D47(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)
                curve_d18O = np.concatenate((model_top_d18O, model_bottom_d18O[1:]))
                curve_D47 = np.concatenate((model_top_D47, model_bottom_D47[1:]))
                curve_time = np.concatenate((tspace1, tspace2[1:]))
                r_curve = recryst_func(theta, curve_time, bcd.t_switch)
            else:
                theta[2] = 0

                model_top_d18O, model_bottom_d18O = DiagenesisModel_d18O_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2)
                model_top_D47, model_bottom_D47 = DiagenesisModel_D47_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)
                curve_d18O = np.concatenate((model_top_d18O, model_bottom_d18O[1:]))
                curve_D47 = np.concatenate((model_top_D47, model_bottom_D47[1:]))
                curve_time = np.concatenate((tspace1, tspace2[1:]))
                r_curve = recryst_func_top(theta, curve_time)

            if time is None:
                time = curve_time
            elif not np.array_equal(time, curve_time):
                raise RuntimeError("Boundary-condition draws returned inconsistent time grids")

            if fit_to == 'd18O':
                curve_noisy_d18O = curve_d18O + rng.normal(0, theta[3], size=curve_d18O.shape)
                curves_d18O.append(curve_noisy_d18O)
                curves_D47.append(curve_D47)
            
            if fit_to == 'D47':
                curve_noisy_D47 = curve_D47 + rng.normal(0, theta[3], size=curve_D47.shape)
                curves_D47.append(curve_noisy_D47)
                curves_d18O.append(curve_d18O)
            
            elif fit_to is None:
                raise RuntimeError("No dataset type indicated, set 'fit_to' as 'd18O' or 'D47' ")

            curves_r.append(r_curve)
            i += 1

        self.tspace = time
        self.theta_curves = theta_curves
        self.d18O_curves = np.asarray(curves_d18O)
        self.D47_curves = np.asarray(curves_D47)
        self.R_curves = np.asarray(curves_r)
        self.SST_curves = np.asarray(curves_SST)

        self.d18O_model16, self.d18O_model50, self.d18O_model84 = np.percentile(self.d18O_curves, [16, 50, 84], axis=0)
        self.d18O_model2_5, _, self.d18O_model97_5 = np.percentile(self.d18O_curves, [2.5, 50, 97.5], axis=0)
        self.D47_model16, self.D47_model50, self.D47_model84 = np.percentile(self.D47_curves, [16, 50, 84], axis=0)
        self.D47_model2_5, _, self.D47_model97_5 = np.percentile(self.D47_curves, [2.5, 50, 97.5], axis=0)        
        self.R_model16, self.R_model50, self.R_model84 = np.percentile(self.R_curves, [16, 50, 84], axis=0)
        self.SST_model2_5, self.SST_model16, self.SST_model50, self.SST_model84, self.SST_model97_5 = np.percentile(self.SST_curves, [2.5, 16, 50, 84, 97.5], axis=0)

        if make_plots:
            self.save_plots(bcd, path)

    def save_plots(self, bcd, path="./"):
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        obs_time = bcd.tspace_data_d18O.cpu().numpy()
        obs_d18O = bcd.d18O.cpu().numpy()
        obs_err_d18O = bcd.d18O_err.cpu().numpy()

        obs_time_D47 = bcd.tspace_data.cpu().numpy()
        obs_D47 = bcd.D47.cpu().numpy()
        obs_err_D47 = bcd.D47_err.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 9))
        ax.fill_betweenx(self.tspace, self.d18O_model2_5, self.d18O_model97_5, color="C0", alpha=0.2, label="95% posterior interval")
        ax.fill_betweenx(self.tspace, self.d18O_model16, self.d18O_model84, color="C0", alpha=0.35, label="68% posterior interval")
        ax.plot(self.d18O_model50, self.tspace, color="C0", label="posterior median")
        ax.errorbar(obs_d18O, obs_time, xerr=obs_err_d18O, fmt="o", color="k", label="data")
        ax.invert_yaxis()
        ax.set(xlabel="d18O carbonate (VSMOW, per mil)", ylabel="Age (Ma)")
        ax.secondary_yaxis("right", functions=(age2depth, depth2age)).set_ylabel("Depth (mbsf)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "d18O_posterior.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 9))
        ax.fill_betweenx(self.tspace, self.D47_model2_5, self.D47_model97_5, color="C0", alpha=0.2, label="95% posterior interval")
        ax.fill_betweenx(self.tspace, self.D47_model16, self.D47_model84, color="C0", alpha=0.35, label="68% posterior interval")
        ax.plot(self.D47_model50, self.tspace, color="C0", label="posterior median")
        ax.errorbar(obs_D47, obs_time_D47, xerr=obs_err_D47, fmt="o", color="k", label="data")
        ax.invert_yaxis()
        ax.set(xlabel="D47 carbonate (per mil)", ylabel="Age (Ma)")
        ax.secondary_yaxis("right", functions=(age2depth, depth2age)).set_ylabel("Depth (mbsf)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "D47_posterior.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 9))
        ax.fill_betweenx(self.tspace, self.R_model16, self.R_model84, color="C1", alpha=0.35)
        ax.plot(self.R_model50, self.tspace, color="C1", label="posterior median")
        ax.invert_yaxis()
        ax.set(xlabel="Recrystallization rate", ylabel="Age (Ma)")
        ax.secondary_yaxis("right", functions=(age2depth, depth2age)).set_ylabel("Depth (mbsf)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "recrystallization_posterior.png", dpi=300)
        plt.close(fig)

        bcd.modelSST.eval()
        bcd.likelihoodSST.eval()
    
        t_min, t_max = bcd.SSTH_time.min().item(), bcd.SSTH_time.max().item()
        query_times = torch.linspace(t_min, t_max, 500)
        query_times_std = (query_times - bcd.SSTH_time.mean()) / bcd.SSTH_time.std()
    
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = bcd.modelSST(query_times_std.to(bcd.SSTH_data.dtype))

            mean = posterior.mean
            lower, upper = posterior.confidence_region()  # +/- 2 std

        mean = mean * bcd.SSTH_data_std + bcd.SSTH_data_mean
        lower = lower * bcd.SSTH_data_std + bcd.SSTH_data_mean
        upper = upper * bcd.SSTH_data_std + bcd.SSTH_data_mean

        fig, ax = plt.subplots(figsize=(8, 9))
        ax.fill_betweenx(query_times.numpy(), lower.numpy(), upper.numpy(), color="C2", alpha=0.35)
        ax.plot(mean.numpy(), query_times.numpy(), color="C2", label="prior median")

        ax.fill_betweenx(self.tspace, self.SST_model2_5, self.SST_model97_5, color="C1", alpha=0.35)
        ax.plot(self.SST_model50, self.tspace, color="C1", label="posterior median")
        ax.invert_yaxis()
        ax.set(xlabel="SST C", ylabel="Age (Ma)")
        ax.secondary_yaxis("right", functions=(age2depth, depth2age)).set_ylabel("Depth (mbsf)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "SST_posterior.png", dpi=300)
        plt.close(fig)

        pd.DataFrame({
            "time_ma": self.tspace,
            "d18O_p2_5": self.d18O_model2_5,
            "d18O_p16": self.d18O_model16,
            "d18O_p50": self.d18O_model50,
            "d18O_p84": self.d18O_model84,
            "d18O_p97_5": self.d18O_model97_5,
        }).to_csv(output_dir / "d18O_posterior_percentiles.csv", index=False)

        pd.DataFrame({
            "time_ma": self.tspace,
            "D47_p2_5": self.D47_model2_5,
            "D47_p16": self.D47_model16,
            "D47_p50": self.D47_model50,
            "D47_p84": self.D47_model84,
            "D47_p97_5": self.D47_model97_5,
        }).to_csv(output_dir / "D47_posterior_percentiles.csv", index=False)

        pd.DataFrame({
            "time_ma": self.tspace,
            "R_p16": self.R_model16,
            "R_p50": self.R_model50,
            "R_p84": self.R_model84,
        }).to_csv(output_dir / "recrystallization_posterior_percentiles.csv", index=False)

        pd.DataFrame({
            "time_ma": self.tspace,
            "SST_p16": self.SST_model16,
            "SST_p50": self.SST_model50,
            "SST_p84": self.SST_model84,
        }).to_csv(output_dir / "SST_posterior_percentiles.csv", index=False)


