import numpy as np
from numba import njit, prange
    
#set up functions for differential equations    
@njit(cache=True)
def initiald18O(SST, seawater_d18O):
    """
    Calculates the initial d18O biogenic carb using SST (SST) and d18O seawater (seawater_d18O) vectors, must be same length/times
    """
    return np.exp((17.57*1000/(273.15+SST)-30.24)/1000)*(1000+seawater_d18O)-1000

@njit(cache=True)
def replace18O(T_deep, seawater_d18O, geotherm): 
    """
    Calculates the newly formed d18O authogenic carb 
    Calculate the sediment tempaertaure using the deep ocean temperature t years after time of deposition (T_deep), the geothermal gradient (geotherm), and the d18O seawater (seawater_d18O) at the start of each parcel's deposition
    """
    temp = T_deep + geotherm -2 # subtract of 2 because that was the assumed deep ocean temperature while constucting the geotherm
    return np.exp((17.57*1000/(273.15+temp)-29.13)/1000)*(1000+seawater_d18O)-1000

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
def diff_eq_d18O_upper(t, y, theta, SST, T_deep, seawater_d18O, geotherm, y0):
    """
    Differential equation calculation for d18O in the upper part of the sediment core.
    Takes in time since deposition (t), current d18O (y), recryst paramaters (theta), SST at time of deposition (SST), 
    depth of parcel (depths), deep ocean temperature t years at time of deposition (T_deep), and d18O seawater (seawater_d18O) at time of deposition, 
    geothermermal gradient (geotherm), and initial d18O (y0)

    """
    R = recryst(t, theta)
    dy = -y0*R + R*replace18O(T_deep, seawater_d18O, geotherm) # preferentially dissolving biogenic carbonate, R integrated should never be over 100%
    return dy

@njit(cache=True)
def diff_eq_d18O_lower(t, y, theta, SST, T_deep, seawater_d18O, geotherm, y0):
    """
    Differential equation calculation for d18O in the lower part of the sediment core
    takes in time since deposition (t), current d18O (y), recryst paramaters (theta), SST at time of deposition (SST), 
    depth of parcel (depths), deep ocean temperature t years at time of deposition (T_deep), d18O seawater (seawater_d18O) at time of deposition,
    geothermermal gradient (geotherm), and initial d18O (y0)
    """
    R = theta[2] # choosing recrystallization to be a constant (alpha), not exponential decay (which would be ~ 0%/Mya this deep)
    dy = -y*R + R*replace18O(T_deep, seawater_d18O, geotherm) # dissolves both biogenic and authogenic carbonate
    return dy

@njit(cache=True)
def diff_eq_d18O_lower_expR(t, y, theta, SST, T_deep, seawater_d18O, geotherm, y0):
    """
    Differential equation calculation for d18O in the lower part of the sediment core
    takes in time since deposition (t), current d18O (y), recryst paramaters (theta), SST at time of deposition (SST), 
    depth of parcel (depths), deep ocean temperature t years at time of deposition (T_deep), d18O seawater (seawater_d18O) at time of deposition,
    geothermermal gradient (geotherm), and initial d18O (y0)
    """
    R = recryst(t, theta) # choosing recrystallization to remain exponential decay at depth
    dy = -y0*R + R*replace18O(T_deep, seawater_d18O, geotherm) # preferentially dissolving biogenic carbonate, R integrated should never be over 100%
    return dy

#set up runge kutta solvers
@njit(cache=True)
def solve_rk45_upper(tspace, theta, y0, T_deep, SST, seawater_d18O, geotherm):
    """
    Explicit RK45 solve for upper core. T_deep, depth, and SST are passed in as vectors to avoid interpolation.
    If tpsace has a step size of h and length n, then:
    SST and seawater_d18O need to have a step size of h and length n
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

        # n = 1 i dne last calc:t = 0; tspace1[:1] - 0 T_deep1[:2] - 0 1 geotherm1[:2] - 0 1 SST1[0] - 0, seawater_d18O1[0] - 0
        # n = 2 i = 0 use:y(0) last calc:t = 1; tspace1[:2] - 0 1 T_deep1[:4] - 0 0.5 1 1.5, geotherm1[:4] - 0 0.5 1 1.5, SST1[1] - 1, seawater_d18O1[1] - 1
        # n = 3 i = 0, 1 use y(1) last calc: t = 2; tspace1[:3] - 0 1 2 T_deep1[:6] - 0 0.5 1 1.5 2 2.5, geotherm1[:6] - 0 0.5 1 1.5 2 2.5, SST1[2] -2, seawater_d18O1[2] - 2

        # solve_rk45_upper(tspace1[:i+1], theta, y0, T_deep1[:2*i+2], depth1[:2*i+2], SST1[i], seawater_d18O1[i], test=i)
        k1 = diff_eq_d18O_upper(t, y, theta, SST, T_deep[-2*i-1], seawater_d18O, geotherm[2*i], y0) 
        k2 = diff_eq_d18O_upper(t + h*0.5, y + h*k1*0.5, theta, SST, T_deep[-2*i-2], seawater_d18O, geotherm[2*i+1], y0)
        k3 = diff_eq_d18O_upper(t + h*0.5, y + h*k2*0.5, theta, SST, T_deep[-2*i-2], seawater_d18O, geotherm[2*i+1], y0)
        k4 = diff_eq_d18O_upper(t + h, y + h*k3, theta, SST, T_deep[-2*i-3], seawater_d18O, geotherm[2*i+2], y0)
        
        y = y + h * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        t = t + h
        
        solution[i + 1] = y
    
    return solution

@njit(cache=True)
def solve_rk45_lower(tspace, theta, y0, T_deep, SST, seawater_d18O, geotherm):
    """
    Explicit RK45 solve for lower core. T_deep, depth, and SST are passed in as vectors to avoid interpolation.
    If tpsace has a step size of h and length n, then:
    SST and seawater_d18O need to have a step size of h and length n
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
        k1 = diff_eq_d18O_lower(t, y, theta, SST, T_deep[-2*i-1], seawater_d18O, geotherm[2*i], y0) 
        k2 = diff_eq_d18O_lower(t + h*0.5, y + h*k1*0.5, theta, SST, T_deep[-2*i-2], seawater_d18O, geotherm[2*i+1], y0)
        k3 = diff_eq_d18O_lower(t + h*0.5, y + h*k2*0.5, theta, SST, T_deep[-2*i-2], seawater_d18O, geotherm[2*i+1], y0)
        k4 = diff_eq_d18O_lower(t + h, y + h*k3, theta, SST, T_deep[-2*i-3], seawater_d18O, geotherm[2*i+2], y0)
        
        y = y + h * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        t = t + h
        
        solution[i + 1] = y
    
    return solution

@njit(cache=True)
def solve_rk45_lower_expR(tspace, theta, y0, T_deep, SST, seawater_d18O, geotherm):
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
        k1 = diff_eq_d18O_lower_expR(t, y, theta, SST, T_deep[-2*i-1], seawater_d18O, geotherm[2*i], y0) 
        k2 = diff_eq_d18O_lower_expR(t + h*0.5, y + h*k1*0.5, theta, SST, T_deep[-2*i-2], seawater_d18O, geotherm[2*i+1], y0)
        k3 = diff_eq_d18O_lower_expR(t + h*0.5, y + h*k2*0.5, theta, SST, T_deep[-2*i-2], seawater_d18O, geotherm[2*i+1], y0)
        k4 = diff_eq_d18O_lower_expR(t + h, y + h*k3, theta, SST, T_deep[-2*i-3], seawater_d18O, geotherm[2*i+2], y0)
        
        y = y + h * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        t = t + h
        
        solution[i + 1] = y
    
    return solution

#set of functions to run the model
@njit(cache=True, parallel=True)
def DiagenesisModel_d18O(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2):
    """
    Run d18O model. Theta conatins the recrystallization parameters, tspace is the times to solve for, T_deep/geotherm/SST/seawater_d18O are
    arrays. Split the arrays tspace/geotherm/SST/seawater_d18O into 1 and 2 based on time to switch differential equations. 
    T_deep1 has the deep ocean temperature for the top part of the core, T_deep2 has the deep ocean temperatures for the whole core.
    If tpsace has a step size of h and length n, then:
    SST and seawater_d18O needs to have a step size of h and length n
    T_deep and geotherm need to have a step size of h/2 and length 2(n+1)
    """
    sol1 = np.zeros_like(tspace1)
    sol2 = np.zeros_like(tspace2)

    len1 = len(T_deep1)

    for i in prange(len(tspace1)):
        y0 = initiald18O(SST1[i], seawater_d18O1[i])
        sol = solve_rk45_upper(tspace1[:i+1], theta, y0, T_deep1[:2*i+1], SST1[i], seawater_d18O1[i], geotherm1[:2*i+1])
        sol1[i] = sol[-1]
    
    for i in prange(len(tspace2)):
        y0 = initiald18O(SST2[i], seawater_d18O2[i]) 
        sol_upper = solve_rk45_upper(tspace1, theta, y0, T_deep2[2*i:2*i+len1], SST2[i], seawater_d18O2[i], geotherm1)
        sol_lower = solve_rk45_lower(tspace2[:i+1], theta, sol_upper[-1], T_deep2[:2*i+1], SST2[i], seawater_d18O2[i], geotherm2[:2*i+1]) 
        sol2[i] = sol_lower[-1]
    return sol1, sol2

@njit(cache=True, parallel=True)
def DiagenesisModel_d18O_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2):
    """
    Run d18O model. Theta conatins the recrystallization parameters, tspace is the times to solve for, T_deep/geotherm/SST/seawater_d18O are
    arrays. Split the arrays T_deep/geotherm/SST/seawater_d18O into 1 and 2 based on time to switch differential equations. 
    T_deep1 has the deep ocean temperature for the top part of the core, T_deep2 has the deep ocean temperatures for the whole core.
    If tpsace has a step size of h and length n, then:
    SST needs to have a step size of h and length n
    T_deep and geotherm need to have a step size of h/2 and length 2(n+1)
    """
    sol1 = np.zeros_like(tspace1)
    sol2 = np.zeros_like(tspace2)

    len1 = len(T_deep1)

    for i in prange(len(tspace1)):
        y0 = initiald18O(SST1[i], seawater_d18O1[i])
        sol = solve_rk45_upper(tspace1[:i+1], theta, y0, T_deep1[:2*i+1], SST1[i], seawater_d18O1[i], geotherm1[:2*i+1])
        sol1[i] = sol[-1]
    
    for i in prange(len(tspace2)):
        y0 = initiald18O(SST2[i], seawater_d18O2[i])
        sol_upper = solve_rk45_upper(tspace1, theta, y0, T_deep2[2*i:2*i+len1], SST2[i], seawater_d18O2[i], geotherm1)
        sol_lower = solve_rk45_lower_expR(tspace2[:i+1], theta, sol_upper[-1], T_deep2[:2*i+1], SST2[i], seawater_d18O2[i], geotherm2[:2*i+1]) 
        sol2[i] = sol_lower[-1]
    return sol1, sol2