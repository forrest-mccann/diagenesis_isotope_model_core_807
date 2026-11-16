import numpy as np
from numba import njit, prange

#set up functions for differential equations
@njit(cache=True)
def cocc_T2D47(T):
    """ 
    Calculate D47 from SST (deg C), used for calculating initial D47 biogenic carb
    """
    return 0.0375*(10**6)/(T+273.15)**2+0.181

@njit(cache=True)
def anderson2T(D47):
    """ 
    Calculate temperature (deg C) from D47
    Uses the clumped temperature conversion defined in Anderson et al. (2021)
    """
    T = 1000*np.sqrt((0.0391)/(D47 - 0.154)) - 273.15
    return T

@njit(cache=True)
def anderson(T):
    """ 
    Calculate D47 from temperature (deg C)
    Uses the clumped temperature conversion defined in Anderson et al. (2021)
    """
    return (0.0391 * 1000000) / ((T + 273.15) ** 2) + 0.154

@njit(cache=True)
def recryst(t, theta):
    """ 
    Paramaterizes recrystallization as exponential decay, uses time (t) and theta = [beta, gamma, alpha, model_s]
    """
    check = theta[0]*theta[1]*(1-np.exp(-1*t/theta[1]))
    if check >= 1: ## ensures that you are not doing more than 100% recrystallization in top section
        return 0
    return theta[0]*np.exp(-t/theta[1])

#set up differential equations
@njit(cache=True)
def diff_eq_D47_upper(t, y, theta, SST, T_deep, geotherm):
    """
    Differential equation calculation for D47 in the upper part of the sediment core
    takes in time since deposition (t), current D47 (y), recryst paramaters (theta), SST at time of deposition (SST), 
    depth of parcel (depth), and deep ocean temperature t years at time of deposition (T_deep), geothermermal gradient (geotherm)
    """
    R = recryst(t, theta) #expoential decay for recrystallization
    y0 = cocc_T2D47(SST)
    env_temp = T_deep + geotherm -2 # subtract of 2 because that was the assumed deep ocean temperature while constucting the geotherm
    replaceD47 = anderson(env_temp)
    dy = -y0*R + R*replaceD47 # preferentially dissolving biogenic carbonate, R integrated should never be over 100%
    
    return dy

@njit(cache=True)
def diff_eq_D47_lower(t, y, theta, SST, T_deep, geotherm):
    """
    Differential equation calculation for D47 in the lower part of the sediment core
    takes in time since deposition (t), current D47 (y), recryst paramaters (theta), SST at time of deposition (SST), 
    depth of parcel (depth), and deep ocean temperature t years at time of deposition (T_deep), and geothermal gradient (geotherm)
    """
    R = theta[2] # choosing recrystallization to be a constant (alpha), not exponential decay (which would be ~ 0%/Mya this deep)
    env_temp = T_deep + geotherm -2 # subtract of 2 because that was the assumed deep ocean temperature while constucting the geotherm
    replaceD47 = anderson(env_temp)
    dy = -y * R + R * replaceD47 # dissolves both biogenic and authogenic carbonate
    return dy

@njit(cache=True)
def diff_eq_D47_lower_expR(t, y, theta, SST, T_deep, geotherm):
    """
    Differential equation calculation for D47 in the lower part of the sediment core
    takes in time since deposition (t), current D47 (y), recryst paramaters (theta), SST at time of deposition (SST), 
    depth of parcel (depth), and deep ocean temperature t years at time of deposition (T_deep), and geothermal gradient (geotherm)
    """
    R = recryst(t, theta) #expoential decay for recrystallization
    y0 = cocc_T2D47(SST)
    env_temp = T_deep + geotherm -2 # subtract of 2 because that was the assumed deep ocean temperature while constucting the geotherm
    replaceD47 = anderson(env_temp)
    dy = -y0 * R + R * replaceD47 # preferentially dissolving biogenic carbonate, R integrated should never be over 100%
    return dy

#set up runge kutta solvers
@njit(cache=True)
def solve_rk45_upper(tspace, theta, y0, T_deep, SST, geotherm):
    """
    Explicit RK45 solve for upper core. T_deep, depth, and SST are passed in as vectors to avoid interpolation.
    If tpsace has a step size of h and length n, then:
    SST needs to have a step size of h and length n
    T_deep and geotherm need to have a step size of h/2 and length 2(n+1)
    """
    n = len(tspace)
    solution = np.zeros(n)
    solution[0] = y0
    
    for i in range(n - 1):
        t = tspace[i]
        y = solution[i]
        h = tspace[i + 1] - t
        
        #RK set up, since method makes calculations at h/2, resolution of geotherm and T_deep need to be 2x of tspace
            # T_deep indexes backwards bc at t=0, the deep ocean T is that of t_final Myr ago
        k1 = diff_eq_D47_upper(t, y, theta, SST, T_deep[-2*i-1], geotherm[2*i]) 
        k2 = diff_eq_D47_upper(t + h*0.5, y + h*k1*0.5, theta, SST, T_deep[-2*i-2], geotherm[2*i+1])
        k3 = diff_eq_D47_upper(t + h*0.5, y + h*k2*0.5, theta, SST, T_deep[-2*i-2], geotherm[2*i+1])
        k4 = diff_eq_D47_upper(t + h, y + h*k3, theta, SST, T_deep[-2*i-3], geotherm[2*i+2])
        
        y = y + h * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        t = t + h
        
        solution[i + 1] = y
    
    return solution

@njit(cache=True)
def solve_rk45_lower(tspace, theta, y0, T_deep, SST, geotherm):
    """
    Explicit RK45 solve for lower core. T_deep, depth, and SST are passed in as vectors to avoid interpolation.
    If tpsace has a step size of h and length n, then:
    SST needs to have a step size of h and length n
    T_deep and geotherm need to have a step size of h/2 and length 2(n+1)
    """
    n = len(tspace)
    solution = np.zeros(n)
    solution[0] = y0
    
    for i in range(n - 1):
        t = tspace[i]
        y = solution[i]
        h = tspace[i + 1] - t
        
        #RK set up, since method makes calculations at h/2, resolution of geotherm and T_deep need to be 2x of tspace
            # T_deep indexes backwards bc at t=0, the deep ocean T is that of t_final Myr ago
        k1 = diff_eq_D47_lower(t, y, theta, SST, T_deep[-2*i-1], geotherm[2*i]) 
        k2 = diff_eq_D47_lower(t + h*0.5, y + h*k1*0.5, theta, SST, T_deep[-2*i-2], geotherm[2*i+1])
        k3 = diff_eq_D47_lower(t + h*0.5, y + h*k2*0.5, theta, SST, T_deep[-2*i-2], geotherm[2*i+1])
        k4 = diff_eq_D47_lower(t + h, y + h*k3, theta, SST, T_deep[-2*i-3], geotherm[2*i+2])
        
        y = y + h * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        t = t + h
        
        solution[i + 1] = y
    
    return solution

@njit(cache=True)
def solve_rk45_lower_expR(tspace, theta, y0, T_deep, SST, geotherm):
    """
    Explicit RK45 solve for lower core. T_deep, depth, and SST are passed in as vectors to avoid interpolation.
    If tpsace has a step size of h and length n, then:
    SST needs to have a step size of h and length n
    T_deep and geotherm need to have a step size of h/2 and length 2(n+1)
    """
    n = len(tspace)
    solution = np.zeros(n)
    solution[0] = y0
    
    for i in range(n - 1):
        t = tspace[i]
        y = solution[i]
        h = tspace[i + 1] - t
        
        #RK set up, since method makes calculations at h/2, resolution of geotherm and T_deep need to be 2x of tspace
            # T_deep indexes backwards bc at t=0, the deep ocean T is that of t_final Myr ago
        k1 = diff_eq_D47_lower_expR(t, y, theta, SST, T_deep[-2*i-1], geotherm[2*i]) 
        k2 = diff_eq_D47_lower_expR(t + h*0.5, y + h*k1*0.5, theta, SST, T_deep[-2*i-2], geotherm[2*i+1])
        k3 = diff_eq_D47_lower_expR(t + h*0.5, y + h*k2*0.5, theta, SST, T_deep[-2*i-2], geotherm[2*i+1])
        k4 = diff_eq_D47_lower_expR(t + h, y + h*k3, theta, SST, T_deep[-2*i-3], geotherm[2*i+2])
        
        y = y + h * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        t = t + h
        
        solution[i + 1] = y
    
    return solution

#set of functions to run the model
@njit(cache=True)
def DiagenesisModel_D47(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2):
    """
    Run D47 model. Theta conatins the recrystallization parameters, tspace is the times to solve for, T_deep/depth/SST/geotherm are
    arrays. Split the arrays tspace/depth/SST/geotherm into 1 and 2 based on time to switch differential equations. 
    T_deep1 has the deep ocean temperature for the top part of the core, T_deep2 has the deep ocean temperatures for the whole core.
    If tpsace has a step size of h and length n, then:
    SST needs to have a step size of h and length n
    T_deep and geotherm need to have a step size of h/2 and length 2(n+1)
    """
    sol1 = np.zeros_like(tspace1)
    sol2 = np.zeros_like(tspace2)

    len1 = len(T_deep1)

    for i in prange(len(tspace1)):
        y0 = cocc_T2D47(SST1[i])
        sol = solve_rk45_upper(tspace1[:i+1], theta, y0, T_deep1[:2*i+1], SST1[i], geotherm1[:2*i+1])
        sol1[i] = sol[-1]
    
    for i in prange(len(tspace2)):
        y0 = cocc_T2D47(SST2[i])
        sol_upper = solve_rk45_upper(tspace1, theta, y0, T_deep2[2*i:2*i+len1], SST2[i], geotherm1)
        sol_lower = solve_rk45_lower(tspace2[:i+1], theta, sol_upper[-1], T_deep2[:2*i+1], SST2[i], geotherm2[:2*i+1]) 
        sol2[i] = sol_lower[-1]
    return sol1, sol2

@njit(cache=True)
def DiagenesisModel_D47_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2):
    """
    Run D47 model. Theta conatins the recrystallization parameters, tspace is the times to solve for, T_deep/SST/geotherm are
    arrays. Split the arrays tspace/SST/geotherm into 1 and 2 based on time to switch differential equations. 
    T_deep1 has the deep ocean temperature for the top part of the core, T_deep2 has the deep ocean temperatures for the whole core.
    If tpsace has a step size of h and length n, then:
    SST needs to have a step size of h and length n
    T_deep and geotherm need to have a step size of h/2 and length 2(n+1)
    """
    sol1 = np.zeros_like(tspace1)
    sol2 = np.zeros_like(tspace2)

    len1 = len(T_deep1)

    for i in prange(len(tspace1)):
        y0 = cocc_T2D47(SST1[i])
        sol = solve_rk45_upper(tspace1[:i+1], theta, y0, T_deep1[:2*i+1], SST1[i], geotherm1[:2*i+1])
        sol1[i] = sol[-1]
    
    for i in prange(len(tspace2)):
        y0 = cocc_T2D47(SST2[i])
        sol_upper = solve_rk45_upper(tspace1, theta, y0, T_deep2[2*i:2*i+len1], SST2[i], geotherm1)
        sol_lower = solve_rk45_lower_expR(tspace2[:i+1], theta, sol_upper[-1], T_deep2[:2*i+1], SST2[i], geotherm2[:2*i+1]) 
        sol2[i] = sol_lower[-1]
    return sol1, sol2
