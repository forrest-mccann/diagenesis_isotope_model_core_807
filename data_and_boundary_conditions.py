import gpytorch, torch
gpytorch.settings.max_cholesky_size._set_value(5000)
import numpy as np
import pandas as pd
from gpytorch.lazy import DiagLazyTensor
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import linregress

# set up requisite functions for boundary condition modelling

df_depth = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='depth-age') #add citation
   
def age2depth(t):
    return np.interp(t, df_depth["age"].values, df_depth["depth"].values)

def depth2age(depth):
    return np.interp(depth, df_depth["depth"].values, df_depth["age"].values)

def to_tensor(series):
    return torch.tensor(series.values, dtype=torch.float32)

def downsample(t, y, n_target=1000):
    df = pd.DataFrame({'t': t, 'y': y}).sort_values('t')
    bins = np.linspace(df.t.min(), df.t.max(), n_target)
    df['bin'] = pd.cut(df.t, bins)
    out = df.groupby('bin', observed=True).agg({'t': 'mean', 'y': 'mean'}).dropna()
    return torch.tensor(out.t.values, dtype=torch.float32), torch.tensor(out.y.values, dtype=torch.float32)

def standardize(y):
    m, s = y.mean(), y.std()
    return (y - m) / s, m, s

def deep_temp_calculator(benthic_d18O, sw_d18O):
    lnalphaCarb_VSMOW1000 = 1000*np.log((1000+benthic_d18O)/(1000+sw_d18O))
    deep_t = 1000*18.03/(lnalphaCarb_VSMOW1000+32.23)-273.15
    return deep_t

# set up functions that make the boundary condition models

# function for geotherm model 
def calculate_geotherm_with_uncertainty(depth_array, porosity_pct_array, k_carb, k_carb_err, q_surface, q_surface_err, T_surface=2.0):
    """
    calculates a geothermal gradient by integrating over thermal conductivity + uncertainty
    takes in matching depth_array (mbsf) and porosity_pct_array (%) arrays, fitted heat flux and thermal conductivity parmeters
    with uncertainty (k_carb, k_carb_err, q_surface, q_surface_err), and the deep ocean temperature present day (T_surface)
    """
    dz = np.diff(depth_array)
    dz = np.append(dz, dz[-1]) 
    
    k_seawater = 0.6 ## citation needed
    porosity_fraction = porosity_pct_array / 100.0
    
    k_bulk = (k_carb ** (1 - porosity_fraction)) * (k_seawater ** porosity_fraction) # use geometric relationship to compute thermal conductivity with depth
    
    # integrate
    dq_integrand = 1.0 / k_bulk
    dkcarb_integrand = -q_surface * (1.0 - porosity_fraction) / (k_carb * k_bulk)
    T_mean = T_surface + np.cumsum((q_surface / k_bulk) * dz)

    #propagate variances to find total uncertainty profile
    S_q = np.cumsum(dq_integrand * dz)
    S_kcarb = np.cumsum(dkcarb_integrand * dz)
    T_variance = (S_q ** 2) * (q_surface_err ** 2) + (S_kcarb ** 2) * (k_carb_err ** 2) # Var(T) = (dT/dq)^2 * Var(q) + (dT/dk_carb)^2 * Var(k_carb)
    T_err = np.sqrt(T_variance)

    T_mean[0] = T_surface
    T_err[0] = 0.0
    
    return T_mean, T_err

# model for fitting our data, d18O or D47
class SingleTaskGPModel(gpytorch.models.ExactGP):
    '''
    GP model to be used with a single time series
    '''
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# make class for importanting data and running all boundary condition models
class import_data_and_make_boundary_conditions:
    '''
    class for importing data, modelling the boundary conditions, and setting the t_switch and resolution (h) of the modelling.
    Reads in excels sheets from a shared folder, has functions for drawing realizations of the boundary conditions + the means for MCMC and MLE fitting
    '''
    def __init__(self, t_switch = 30, h = 0.5):
            self.h = h
            self.t_switch = t_switch

            # import our measured + lit data

            df_D47_turner = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='D47 Turner data')
            df_D47_stolper = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='D47 Stolper 2018 data')

            D47_turner_data = df_D47_turner["D47"].values
            D47_turner_data_err = df_D47_turner["D47_err"].values
            D47_turner_depth_data = df_D47_turner["depth"].values
            D47_turner_age_data = depth2age(D47_turner_depth_data)

            D47_stolper_data = df_D47_stolper["D47"].values
            D47_stolper_data_err = df_D47_stolper["D47_err"].values
            D47_stolper_depth_data = df_D47_stolper["depth"].values
            D47_stolper_age_data = depth2age(D47_stolper_depth_data)


            tspace_data = np.concatenate([D47_turner_age_data, D47_stolper_age_data])
            D47 = np.concatenate([D47_turner_data, D47_stolper_data])
            D47_err = np.concatenate([D47_turner_data_err, D47_stolper_data_err])

            sort_indices = np.argsort(tspace_data)

            df_d18O_turner = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='d18O Turner data')
            df_d18O_stolper = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='d18O Stolper 2018 data')

            d18O_turner_data = df_d18O_turner["d18O_carb (VSMOW)"].values
            d18O_turner_data_err = df_d18O_turner["d18O_err"].values
            d18O_turner_depth_data = df_d18O_turner["depth"].values
            d18O_turner_age_data = depth2age(d18O_turner_depth_data)

            d18O_stolper_data = df_d18O_stolper["d18O_carb (VSMOW)"].values
            d18O_stolper_data_err = df_d18O_stolper["d18O_err"].values
            d18O_stolper_depth_data = df_d18O_stolper["depth"].values
            d18O_stolper_age_data = depth2age(d18O_stolper_depth_data)

            tspace_data_d18O = np.concatenate([d18O_turner_age_data, d18O_stolper_age_data])
            d18O = np.concatenate([d18O_turner_data, d18O_stolper_data])
            d18O_err = np.concatenate([d18O_turner_data_err, d18O_stolper_data_err])

            sort_indices = np.argsort(tspace_data_d18O)

            tspace_data_d18O = tspace_data_d18O[sort_indices]
            d18O = d18O[sort_indices]
            d18O_err = d18O_err[sort_indices] 

            # import ODP data
            imported_data = {}

            root_dir = "."
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    # look for Excel files and ignore temporary/hidden lock files
                    if file.endswith(".xlsx") and not file.startswith("~$"):
                        
                        # clean up file name
                        file_name_clean = file.replace('.xlsx', '')
                        
                        # get full path to the file for reading
                        file_path = os.path.join(root, file)
                        
                        # import all sheets
                        try:
                            excel_dict = pd.read_excel(file_path, sheet_name=None)
                            
                            for sheet_name, df in excel_dict.items():
                                # Construct the custom key name: "Filename - Sheetname"
                                dict_key = f"{file_name_clean} - {sheet_name}"
                                
                                imported_data[dict_key] = df
                                print(f"Imported: {dict_key}")
                                
                        except Exception as e:
                            print(f"Could not read {file_path}: {e}")

            print(f"\nTotal imported: {len(imported_data)}")

            keys_to_plot = ["806", "805", "804", "803", "1183", "1184", "1185", "1186", "1187"]
            markers = { 
                "804": "o", "803": "^", "805": "s", "806": "D", 
                "1183": "d", "1184": "v", "1185": "8", "1186": "p", "1187": "h",
            }

            active_caco3_values = []
            all_porosity = []
            all_thermcon = []

            for key in keys_to_plot:
                sheet_key = f"leg130and192data - {key}"
                if sheet_key not in imported_data: continue
                df_sheet = imported_data[sheet_key]
                site_cores = [f"{key}A", f"{key}B", f"{key}C", f"{key}D"]
                for core in site_cores:
                    if "SITE+CORE+SECT" in df_sheet.columns and "CaCO3 (%)" in df_sheet.columns and "Thermcon Value" in df_sheet.columns:
                        core_df = df_sheet[df_sheet["SITE+CORE+SECT"].astype(str).str.contains(core)]
                        valid_rows = core_df.dropna(subset=["Thermcon Value", "Porosity", "CaCO3 (%)"])
                        active_caco3_values.extend(valid_rows["CaCO3 (%)"].tolist())
                        all_porosity.extend(valid_rows["Porosity"].tolist())
                        all_thermcon.extend(valid_rows["Thermcon Value"].tolist())

            # run fitting for geothermal gradient boundary condition
                # take measured thermal conductivty data for nearby sites and create a porosity-thermal conductivity relationship
                # take measured temperatures and determine a surface heat flux
                # use the heat flux and porosity-thermal conductivity relationship and integrate over 807 porosity to get a geotherm
            
            thermcond_seawater = 0.6  

            def geometric_mean_model(porosity_pct, t_carb):
                porosity_fraction = porosity_pct / 100.0
                return (t_carb ** (1 - porosity_fraction)) * (thermcond_seawater ** porosity_fraction)

            # take measured thermal conductivty data for nearby sites and create a porosity-thermal conductivity relationship
            popt, pcov = curve_fit(geometric_mean_model, all_porosity, all_thermcon, p0=[4.0])
            thermcond_carb = popt[0]
            b_variance = pcov[0, 0]
            thermcond_carb_error = np.sqrt(b_variance)

            # take measured temperatures and determine a surface heat flux
            array = np.array([[1.995732574679943, 0.05328596802841368],[4.570412517780938, 44.56483126110125],[5.544807965860597, 63.71225577264656],[6.248933143669985, 82.7353463587922]])
            U1330B = pd.DataFrame(array,columns=['Temp (°C)','Depth (mbsf)'])
            x_depths = U1330B["Depth (mbsf)"]
            y_temps = U1330B["Temp (°C)"]

            regression_result = linregress(x_depths, y_temps)
            slope2 = regression_result.slope 
            intercept2 = regression_result.intercept  
            geotherm_err = (regression_result.stderr) 

            depths = np.array(U1330B["Depth (mbsf)"])
            intervals = np.diff(np.insert(depths, 0, 0.0))
            depth_target = max(depths)

            U1330B_porosity = -0.0377 * depths + 71.167
            porosity_fraction = U1330B_porosity / 100.0
            thermcond_seawater = 0.6

            n_samples = 5000
            sampled_k_carb = np.random.normal(thermcond_carb, thermcond_carb_error, size=(n_samples, 1))
            sampled_geotherm = np.random.normal(slope2, geotherm_err, size=n_samples)
            sampled_conductivities = (sampled_k_carb ** (1 - porosity_fraction)) * (thermcond_seawater**porosity_fraction)

            thermal_resistance = np.sum(intervals / sampled_conductivities, axis=1)
            k_eff_distribution = depth_target / thermal_resistance
            q_surf_distribution = sampled_geotherm * k_eff_distribution

            q_surface_mean = np.mean(q_surf_distribution)
            q_surface_std = np.std(q_surf_distribution)

            # use the heat flux and porosity-thermal conductivity relationship and integrate over 807 porosity to get a geotherm
            df = pd.read_csv(f"site_807_depth_porosity.csv")
            data_depths_m = np.asarray(df['Depth (mbsf)'].tolist())
            data_porosity = np.asarray(df['Porosity (%)'].tolist())

            self.geotherm_mean, self.geotherm_err = calculate_geotherm_with_uncertainty(depth_array=data_depths_m, porosity_pct_array=data_porosity, k_carb=thermcond_carb, k_carb_err=thermcond_carb_error, q_surface=q_surface_mean, q_surface_err=q_surface_std, T_surface=2.0)
            self.geotherm_upper_95 = self.geotherm_mean + 1.96 * self.geotherm_err
            self.geotherm_lower_95 = self.geotherm_mean - 1.96 * self.geotherm_err
            self.geotherm_depths = data_depths_m
            self.sampled_k_carb = sampled_k_carb      
            self.sampled_geotherm = sampled_geotherm
            self.q_surf_distribution = q_surf_distribution
            self.geotherm_depths_full = data_depths_m
            self.data_porosity_full = data_porosity

            # run fitting for measured + lit data
                # Gaussian process for 807 D47
                # Gaussian process for 807 d18O

            # Gaussian process for 807 D47
            self.D47 = torch.from_numpy(D47) 
            self.D47_err = torch.from_numpy(D47_err)
            self.tspace_data = torch.from_numpy(tspace_data)

            tspace_data_norm = (self.tspace_data - self.tspace_data.mean()) / self.tspace_data.std()
            D47_norm, self.D47_mean, self.D47_std = standardize(self.D47)
            D47_noise_var = (D47_err / self.D47_std) ** 2  

            self.likelihoodD47 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=D47_noise_var.double(), learn_additional_noise=True)
            self.modelD47 = SingleTaskGPModel(tspace_data_norm.double(), D47_norm.double(), self.likelihoodD47)

            self.modelD47.train(); self.likelihoodD47.train()

            optimizer = torch.optim.Adam(self.modelD47.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoodD47, self.modelD47)
            for it in range(500):
                optimizer.zero_grad()
                output = self.modelD47(tspace_data_norm.double())
                loss = -mll(output, D47_norm.double())
                loss.backward()
                optimizer.step()
                if it % 50 == 0:
                    print(f"iter {it}  loss {loss.item():.3f}")
            

            # Gaussian process for 807 d18O
            self.d18O = torch.from_numpy(d18O) 
            self.d18O_err = torch.from_numpy(d18O_err)
            self.tspace_data_d18O = torch.from_numpy(tspace_data_d18O)

            tspace_data_norm_d18O = (self.tspace_data_d18O - self.tspace_data_d18O.mean()) / self.tspace_data_d18O.std()
            d18O_norm, self.d18O_mean, self.d18O_std = standardize(self.d18O)

            d18O_noise_var = (d18O_err / self.d18O_std) ** 2  

            self.likelihoodd18O = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=d18O_noise_var.double(), learn_additional_noise=True)
            self.modeld18O = SingleTaskGPModel(tspace_data_norm_d18O.double(), d18O_norm.double(), self.likelihoodd18O)

            self.modeld18O.train(); self.likelihoodd18O.train()
            optimizer = torch.optim.Adam(self.modeld18O.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoodd18O, self.modeld18O)
            for it in range(500):
                optimizer.zero_grad()
                output = self.modeld18O(tspace_data_norm_d18O.double())
                loss = -mll(output, d18O_norm.double())
                loss.backward()
                optimizer.step()
                if it % 50 == 0:
                    print(f"iter {it}  loss {loss.item():.3f}") 

            # run GP for boundary conditions SST and benthic d18O
            # import seawater d18O model (rholing) and adjust to right time grid

            SSTH = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='SST data')
            SSTH = SSTH.sort_values(by='age', ascending=True)
            SSTH_time = SSTH['age']
            SSTH_data = SSTH['SST']
        
            benthic_d18O = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='Benthic Foram d18O')
            benthic_d18O = benthic_d18O.sort_values(by='age', ascending=True)
            benthic_time = benthic_d18O['age']
            benthic_d18O_data = benthic_d18O['d18O_carb VSMOW']
               

            seawater_d18O_data = pd.read_excel('./data_and_boundary_conditions.xlsx', sheet_name='seawater 18O')
            seawater_d18O_data = seawater_d18O_data.sort_values(by='age', ascending=True)
            seawater_d18O_time = seawater_d18O_data['age']
            seawater_d18O_data_data = seawater_d18O_data['Ox_18']
            

            SSTH_time, SSTH_data = downsample(SSTH_time, SSTH_data, n_target=500)
            self.SSTH_time = SSTH_time
            self.SSTH_data = SSTH_data
            
            SSTH_time_norm = (SSTH_time - SSTH_time.mean()) / SSTH_time.std()
            SSTH_data_norm, SSTH_data_mean, SSTH_data_std = standardize(SSTH_data)
            self.SSTH_data_mean = SSTH_data_mean
            self.SSTH_data_std = SSTH_data_std

            self.likelihoodSST = gpytorch.likelihoods.GaussianLikelihood().double()
            self.modelSST = SingleTaskGPModel(SSTH_time_norm.double(), SSTH_data_norm.double(), self.likelihoodSST).double()

            self.modelSST.train(); self.likelihoodSST.train()

            optimizer = torch.optim.Adam(self.modelSST.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoodSST, self.modelSST)
            for it in range(500):
                optimizer.zero_grad()
                output = self.modelSST(SSTH_time_norm.double())
                loss = -mll(output, SSTH_data_norm.double())
                loss.backward()
                optimizer.step()
                if it % 50 == 0:
                    print(f"iter {it}  loss {loss.item():.3f}")
            
            benthic_time, benthic_d18O_data = downsample(benthic_time, benthic_d18O_data, n_target=500)
            self.benthic_time = benthic_time
            self.benthic_d18O_data = benthic_d18O_data

            benthic_time_norm = (benthic_time - benthic_time.mean()) / benthic_time.std()
            benthic_d18O_norm, benthic_d18O_mean, benthic_d18O_std = standardize(benthic_d18O_data)
            self.benthic_d18O_mean = benthic_d18O_mean 
            self.benthic_d18O_std = benthic_d18O_std
                

            self.likelihood_benthic = gpytorch.likelihoods.GaussianLikelihood().double()
            self.model_benthic = SingleTaskGPModel(benthic_time_norm.double(), benthic_d18O_norm.double(), self.likelihood_benthic).double()

            self.model_benthic.train(); self.likelihood_benthic.train()

            optimizer = torch.optim.Adam(self.model_benthic.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_benthic, self.model_benthic)
            for it in range(500):
                optimizer.zero_grad()
                output = self.model_benthic(benthic_time_norm.double())
                loss = -mll(output, benthic_d18O_norm.double())
                loss.backward()
                optimizer.step()
                if it % 50 == 0:
                    print(f"iter {it}  loss {loss.item():.3f}")

    
            # create two time vectors to be compatible with RK45 numerical method
            self.coarse_time = torch.arange(0.0, 77.0 + h / 2, h, dtype=torch.float64)
            self.fine_time = torch.arange(0.0, 77.0 + h / 4, h / 2, dtype=torch.float64)

            self.seawater_fine = np.interp(self.fine_time, seawater_d18O_time, seawater_d18O_data_data)
            self.seawater_coarse = np.interp(self.coarse_time, seawater_d18O_time, seawater_d18O_data_data)

            assert torch.allclose(self.fine_time[::2], self.coarse_time)
            assert np.isclose(self.t_switch / h, round(self.t_switch / h))

    # functions to save boundary condition models (avoid retraining GPs)and load in boundary conditions from a save file
    def save_boundary_conditions(self, filepath):
        '''
        saves everything to describe the SSTH/benthic foram/sw 18O distribution, geotherm fitting, and d18O/D47 data fitting as a dictionary
        '''
        checkpoint = {
            'h': self.h,
            't_switch': self.t_switch,
            'coarse_time': self.coarse_time,
            'fine_time': self.fine_time,
 
            # geotherm boundary condition
            'geotherm_mean': self.geotherm_mean,
            'geotherm_err': self.geotherm_err,
            'geotherm_upper_95': self.geotherm_upper_95,
            'geotherm_lower_95': self.geotherm_lower_95,
            'geotherm_depths': self.geotherm_depths,
            'geotherm_depths_full': self.geotherm_depths_full,
            'data_porosity_full': self.data_porosity_full,
            'sampled_k_carb': self.sampled_k_carb,
            'sampled_geotherm': self.sampled_geotherm,
            'q_surf_distribution': self.q_surf_distribution,
 
            # D47 GP
            'D47': self.D47,
            'D47_err': self.D47_err,
            'tspace_data': self.tspace_data,
            'D47_mean': self.D47_mean,
            'D47_std': self.D47_std,
            'modelD47_state_dict': self.modelD47.state_dict(),
            'likelihoodD47_state_dict': self.likelihoodD47.state_dict(),
 
            # d18O GP
            'd18O': self.d18O,
            'd18O_err': self.d18O_err,
            'tspace_data_d18O': self.tspace_data_d18O,
            'd18O_mean': self.d18O_mean,
            'd18O_std': self.d18O_std,
            'modeld18O_state_dict': self.modeld18O.state_dict(),
            'likelihoodd18O_state_dict': self.likelihoodd18O.state_dict(),

            # SST GP
            'SSTH': self.SSTH_data,
            'SSTH_time': self.SSTH_time,
            'SSTH_data_mean': self.SSTH_data_mean, 
            'SSTH_data_std': self.SSTH_data_std,
            'modelSST_state_dict': self.modelSST.state_dict(),
            'likelihoodSST_state_dict': self.likelihoodSST.state_dict(),

            # benthic d18O GP
            'benthic_d18O': self.benthic_d18O_data,
            'benthic_time': self.benthic_time,
            'benthic_d18O_mean': self.benthic_d18O_mean, 
            'benthic_d18O_std': self.benthic_d18O_std,
            'model_benthic_state_dict': self.model_benthic.state_dict(),
            'likelihood_benthic_state_dict': self.likelihood_benthic.state_dict(),

            # seawater d18O model
            'seawater_d18O_fine': self.seawater_fine,
            'seawater_d18O_coarse': self.seawater_coarse,

        }
        torch.save(checkpoint, filepath)
        print(f"Saved boundary condition distributions to {filepath}")
 
    @classmethod
    def load_boundary_conditions(cls, filepath):
        '''
        rebuilds a fully-fitted import_data_and_make_boundary_conditions object from a file written by save_boundary_conditions().
        '''
        checkpoint = torch.load(filepath, weights_only=False)

        obj = cls.__new__(cls)
 
        obj.h = checkpoint['h']
        obj.t_switch = checkpoint['t_switch']
        obj.coarse_time = checkpoint['coarse_time']
        obj.fine_time = checkpoint['fine_time']
 
        obj.geotherm_mean = checkpoint['geotherm_mean']
        obj.geotherm_err = checkpoint['geotherm_err']
        obj.geotherm_upper_95 = checkpoint['geotherm_upper_95']
        obj.geotherm_lower_95 = checkpoint['geotherm_lower_95']
        obj.geotherm_depths = checkpoint['geotherm_depths']
        obj.geotherm_depths_full = checkpoint['geotherm_depths_full']
        obj.data_porosity_full = checkpoint['data_porosity_full']
        obj.sampled_k_carb = checkpoint['sampled_k_carb']
        obj.sampled_geotherm = checkpoint['sampled_geotherm']
        obj.q_surf_distribution = checkpoint['q_surf_distribution']
 
        obj.D47 = checkpoint['D47']
        obj.D47_err = checkpoint['D47_err']
        obj.tspace_data = checkpoint['tspace_data']
        obj.D47_mean = checkpoint['D47_mean']
        obj.D47_std = checkpoint['D47_std']
 
        tspace_data_norm = (obj.tspace_data - obj.tspace_data.mean()) / obj.tspace_data.std()
        D47_norm = (obj.D47 - obj.D47_mean) / obj.D47_std
        D47_noise_var = (obj.D47_err / obj.D47_std) ** 2
 
        obj.likelihoodD47 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=D47_noise_var.double(), learn_additional_noise=True)
        obj.modelD47 = SingleTaskGPModel(tspace_data_norm.double(), D47_norm.double(), obj.likelihoodD47)
        obj.modelD47.load_state_dict(checkpoint['modelD47_state_dict'])
        obj.likelihoodD47.load_state_dict(checkpoint['likelihoodD47_state_dict'])
 
        obj.d18O = checkpoint['d18O']
        obj.d18O_err = checkpoint['d18O_err']
        obj.tspace_data_d18O = checkpoint['tspace_data_d18O']
        obj.d18O_mean = checkpoint['d18O_mean']
        obj.d18O_std = checkpoint['d18O_std']
 
        tspace_data_norm_d18O = (obj.tspace_data_d18O - obj.tspace_data_d18O.mean()) / obj.tspace_data_d18O.std()
        d18O_norm = (obj.d18O - obj.d18O_mean) / obj.d18O_std
        d18O_noise_var = (obj.d18O_err / obj.d18O_std) ** 2
 
        obj.likelihoodd18O = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=d18O_noise_var.double(), learn_additional_noise=True).double()
        obj.modeld18O = SingleTaskGPModel(tspace_data_norm_d18O.double(), d18O_norm.double(), obj.likelihoodd18O).double()
        obj.benthic_d18O_mean = checkpoint['benthic_d18O_mean']
        obj.benthic_d18O_std = checkpoint['benthic_d18O_std']
        obj.modeld18O.load_state_dict(checkpoint['modeld18O_state_dict'])
        obj.likelihoodd18O.load_state_dict(checkpoint['likelihoodd18O_state_dict'])


        obj.SSTH_data = checkpoint['SSTH']
        obj.SSTH_time = checkpoint['SSTH_time']
        obj.SSTH_data_mean = checkpoint['SSTH_data_mean'] 
        obj.SSTH_data_std = checkpoint['SSTH_data_std']

        SSTH_time_norm = (obj.SSTH_time - obj.SSTH_time.mean()) / obj.SSTH_time.std()
        SSTH_data_norm, SSTH_data_mean, SSTH_data_std = standardize(obj.SSTH_data)
            

        obj.likelihoodSST = gpytorch.likelihoods.GaussianLikelihood().double()
        obj.modelSST = SingleTaskGPModel(SSTH_time_norm.double(), SSTH_data_norm.double(), obj.likelihoodSST).double()
  
        obj.modelSST.load_state_dict(checkpoint['modelSST_state_dict'])
        obj.likelihoodSST.load_state_dict(checkpoint['likelihoodSST_state_dict'])

        obj.benthic_d18O_data = checkpoint['benthic_d18O']
        obj.benthic_time = checkpoint['benthic_time']

        benthic_time_norm = (obj.benthic_time - obj.benthic_time.mean()) / obj.benthic_time.std()
        benthic_d18O_data_norm, benthic_d18O_data_mean, benthic_d18O_data_std = standardize(obj.benthic_d18O_data)
            
        obj.likelihood_benthic = gpytorch.likelihoods.GaussianLikelihood()
        obj.model_benthic = SingleTaskGPModel(benthic_time_norm.double(), benthic_d18O_data_norm.double(), obj.likelihood_benthic)
    
        obj.model_benthic.load_state_dict(checkpoint['model_benthic_state_dict'])
        obj.likelihood_benthic.load_state_dict(checkpoint['likelihood_benthic_state_dict'])

        obj.seawater_fine = checkpoint['seawater_d18O_fine']
        obj.seawater_coarse = checkpoint['seawater_d18O_coarse']
 
 
        print(f"Loaded boundary condition distributions from {filepath}")
        return obj

    # with boundary condition modelling complete, define functions that draw plasible boundary conditions from posterior distributions
    # BC1: SSTh, BC2: benthic foram d18O, BC3: bottom ocean temperature, BC4: seawater d18O
        # draw geotherm boundary condition
        # caching function for LCM/GP joint boundary conditions
        # draw BC1, 2, and 4 jointly
        # use draw geotherm and draw BC124 to make draw all boundary conditions
        # draw mean joint BC124 (for MLE calcs and plotting)
        # draw mean all BCs

    # draw geotherm boundary condition
    def draw_geotherm_batch(self, query_depths, rng, n_draws):
        '''
        draw n_draws realizations of the geotherm boundary condition at once
        '''

        m = rng.integers(0, len(self.sampled_k_carb), size=n_draws)
        k_carb = self.sampled_k_carb[m, 0][:, None]     
        q_surf = self.q_surf_distribution[m][:, None]   

        phi = np.interp(query_depths, self.geotherm_depths_full, self.data_porosity_full) / 100.0   
        dz = np.diff(query_depths, prepend=query_depths[0])

        k_bulk = k_carb ** (1 - phi) * 0.6 ** phi
        return 2.0 + np.cumsum(q_surf / k_bulk * dz, axis=1)
    
    def _ensure_posterior_cached(self):
        '''
        saves the posterior distribution for the single-task GP model
        '''
        if hasattr(self, "_posterior_cached"):
            return

        all_times = torch.unique(self.fine_time, sorted=True)
        self._all_times = all_times
        all_times_std = (all_times - self.benthic_time.mean()) / self.benthic_time.std()

        self.model_benthic.double().eval()
        self.likelihood_benthic.double().eval()   
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model_benthic(all_times_std.double())
            self._posterior_cached = posterior

    def _ensure_eigen_decomp_cached(self, var_frac=0.75): #### change var_fraction if you want to change the expansion size
        '''
        saves the KL expansion of the SST GP posterior
        '''
        if hasattr(self, "_SST_cached"):
            return
        t = self.coarse_time
        t_std = ((t - self.SSTH_time.mean()) / self.SSTH_time.std()).double()

        self.modelSST.eval()
        with torch.no_grad():
            posterior = self.modelSST(t_std)
            mean = posterior.mean
            cov = posterior.covariance_matrix.double()
        cov = 0.5 * (cov + cov.T)

        evals, evecs = torch.linalg.eigh(cov)
        evals, evecs = evals.flip(0).clamp_min(0.0), evecs.flip(1)

        frac = torch.cumsum(evals, 0) / evals.sum()
        K = int(torch.searchsorted(frac, var_frac)) + 1

        self._SST_cached = dict(
            t=t, mean=mean,
            evals=evals[:K],
            evecs=evecs[:, :K],        # (n_grid, K), orthonormal columns
            basis=evecs[:, :K] * evals[:K].sqrt(),   # (n_grid, K), the "scaled modes"
        )

    # use draw geotherm and draw BC124 to make draw all boundary conditions
    def draw_all_BCs_batch(self, n_draws, rng):
        """
        Draw n_draws from SSTH, benthic foram d18O, seawater d18O and geotherm boundary-condition distributions
        calculates bottom water temperature for each draw too
        """
        n_draws = int(n_draws)
        if n_draws < 1:
            raise ValueError("n_draws must be at least one")

        coarse_full = self.coarse_time
        fine_full = self.fine_time

        i_coarse = torch.searchsorted(coarse_full, torch.tensor(self.t_switch, dtype=coarse_full.dtype), right=True)
        i_fine = torch.searchsorted(fine_full, torch.tensor(self.t_switch, dtype=fine_full.dtype), right=True)

        self._ensure_posterior_cached()
        self._ensure_eigen_decomp_cached()

        with torch.no_grad():
            benthic_samples_norm = self._posterior_cached.rsample(torch.Size([n_draws]))

        samples_benthic = benthic_samples_norm*self.benthic_d18O_std + self.benthic_d18O_mean

        SST_decomp = self._SST_cached
        SST_basis  = SST_decomp["basis"].cpu().numpy()
        SST_mean_norm = SST_decomp["mean"].cpu().numpy()

        sw_fine = np.tile(self.seawater_fine,(n_draws,1))
        sw_coarse_full = np.tile(self.seawater_coarse,(n_draws,1))
        
        deep_T_full = deep_temp_calculator(samples_benthic.cpu().numpy(), sw_fine) ##########here
        depth_full = age2depth(fine_full.cpu().numpy())

        geotherm_full = self.draw_geotherm_batch(depth_full, rng, n_draws)

        return (
            # Coarse time grids: shape (n_coarse_segment,)
            coarse_full[:i_coarse].cpu().numpy(),
            coarse_full[i_coarse - 1:].cpu().numpy(),

            # Fine-grid deep temperatures: shape (n_draws, n_fine_segment)
            deep_T_full[:, :i_fine],
            deep_T_full, 

            # Fine depth grids: shape (n_fine_segment,)
            depth_full[:i_fine],
            depth_full[i_fine - 1:],

            # Coarse SST: shape (n_draws, n_coarse_segment)  ##### change this
            SST_basis,
            SST_mean_norm,
            self.SSTH_data_mean.cpu().numpy(), 
            self.SSTH_data_std.cpu().numpy(),

            # Coarse seawater d18O: shape (n_draws, n_coarse_segment)
            self.seawater_coarse[:i_coarse],
            self.seawater_coarse[i_coarse - 1:],

            # Fine geotherms: shape (n_draws, n_fine_segment)
            geotherm_full[:, :i_fine],
            geotherm_full[:, i_fine - 1:],

            np.array([i_fine, i_coarse]),
        )

    # draw mean all BCs
    def mean_all_BCs(self):
        '''
        Draw the mean for SSTH, benthic foram d18O, seawater d18O, and geotherm from the boundary condition distributions, use for MLE fitting or plotting
        Also calculates deep ocean temperature
        '''
        coarse_full = self.coarse_time
        fine_full = self.fine_time

        i_coarse = torch.searchsorted(coarse_full, torch.tensor(self.t_switch, dtype=coarse_full.dtype), right=True)
        i_fine = torch.searchsorted(fine_full, torch.tensor(self.t_switch, dtype=fine_full.dtype), right=True)

        self._ensure_posterior_cached()
        self._ensure_eigen_decomp_cached()

        with torch.no_grad():
            benthic_mean_norm = self._posterior_cached.mean

        mean_benthic = benthic_mean_norm*self.benthic_d18O_std + self.benthic_d18O_mean

        SST_decomp = self._SST_cached
        SST_mean_norm = SST_decomp["mean"].cpu().numpy()
        SST_mean = SST_mean_norm*self.SSTH_data_std.cpu().numpy() + self.SSTH_data_mean.cpu().numpy()
        
        deep_T = deep_temp_calculator(mean_benthic.cpu().numpy(), self.seawater_fine)
        depth = age2depth(fine_full.cpu().numpy())

        geotherm = np.interp(depth, self.geotherm_depths, self.geotherm_mean)

        return (
            coarse_full[:i_coarse].cpu().numpy(),
            coarse_full[i_coarse - 1:].cpu().numpy(),

            deep_T[:i_fine],
            deep_T,

            depth[:i_fine],
            depth[i_fine - 1:],

            SST_mean[:i_coarse],
            SST_mean[i_coarse - 1:],

            self.seawater_coarse[:i_coarse],
            self.seawater_coarse[i_coarse - 1:],

            geotherm[:i_fine],
            geotherm[i_fine - 1:],
        )
        

# some plotting functions to visualize the boundary condition modelling                

def plot_single_gp_fit(model, likelihood, t_obs, y_obs, x_mean, x_std, y_mean, y_std, name="d18O", n_query=200, include_obs_noise=False):
    """
    plot GP fit to d18O or D47
    """
    model.eval()
    likelihood.eval()
 
    t_min, t_max = t_obs.min().item(), t_obs.max().item()
    query_times = torch.linspace(t_min, t_max, n_query)
    query_times_std = (query_times - x_mean) / x_std
 
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(query_times_std.to(y_obs.dtype))
        if include_obs_noise:
            posterior = likelihood(posterior)
        mean = posterior.mean
        lower, upper = posterior.confidence_region()  # +/- 2 std

    mean = mean * y_std + y_mean
    lower = lower * y_std + y_mean
    upper = upper * y_std + y_mean
 
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(query_times.numpy(), mean.numpy(), color="C0", label="posterior mean")
    ax.fill_between(query_times.numpy(), lower.numpy(), upper.numpy(),
                     color="C0", alpha=0.25, label="95% CI")
    ax.scatter(t_obs.numpy(), y_obs.numpy(), color="k", s=20, zorder=5, label="observations")
    ax.set_ylabel(name)
    ax.set_xlabel("time")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"GP fit to {name} data")
    fig.tight_layout()
    # save mean/lower/upper to CSV
    safe_name = name.replace(" ", "_")
    df = pd.DataFrame({
        "time": query_times.numpy(),
        "mean": mean.numpy(),
        "lower": lower.numpy(),
        "upper": upper.numpy(),
    })
    df.to_csv(f"./{safe_name}_GP_fit.csv", index=False)

    return fig, ax

            