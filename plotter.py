import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from diagenesis_model_d18O import DiagenesisModel_d18O, DiagenesisModel_d18O_expR, recryst
from diagenesis_model_D47 import DiagenesisModel_D47, DiagenesisModel_D47_expR
from diagenesis_model_D47 import anderson2T
from data_and_boundary_conditions import age2depth, depth2age

def plot_D47(theta = None, boundary_conditions_and_data = None):
    """
    Function will default plot the D47 data. Given a theta = [beta, gamma, alpha, model_s], it will plot the mean model output.
    """
    bcd = boundary_conditions_and_data

    fig, ax = plt.subplots(figsize=(10, 10))
    if theta is not None:
        theta = np.asarray(theta)
        (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()
        model_output = DiagenesisModel_D47(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)
        ax.plot(model_output[0], tspace1, label = f'top model-theta {np.round(theta, decimals=4)}')
        ax.plot(model_output[1], tspace2, label = f'bottom model-theta {np.round(theta, decimals=4)}')


    ax.errorbar(bcd.D47, bcd.tspace_data, xerr=bcd.D47_err, fmt="o", color="k", label="stolper + turner data")
    ax.invert_yaxis()
    ax.set_title('D47 v. age/depth')
    ax.set_xlabel('D47 permil')
    ax.set_ylabel('Age (Mya)')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()

def plot_T_clump(theta = None, boundary_conditions_and_data = None):
    """
    Function will default plot the T_clump data. Given a theta = [beta, gamma, alpha, model_s], it will plot the mean model output.
    """
    bcd = boundary_conditions_and_data
    fig, ax = plt.subplots(figsize=(10, 10))
    if theta is not None:
        theta = np.asarray(theta)
        (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()
        model_output = DiagenesisModel_D47(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)
        ax.plot(anderson2T(model_output[0]), tspace1, label = f'top model-theta {np.round(theta, decimals=4)}')
        ax.plot(anderson2T(model_output[1]), tspace2, label = f'bottom model-theta {np.round(theta, decimals=4)}')


    ax.errorbar(anderson2T(bcd.D47), bcd.tspace_data, xerr=bcd.D47_err, fmt="o", color="k", label="stolper + turner data")
    ax.invert_yaxis()
    ax.set_title('T_clump v. age/depth')
    ax.set_xlabel('Temperature (deg C)')
    ax.set_ylabel('Age (Mya)')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()

def plot_d18O(theta = None, boundary_conditions_and_data = None):
    """
    Function will default plot the d18O data. Given a theta = [beta, gamma, alpha], it will plot a model. 
    """
    bcd = boundary_conditions_and_data
    fig, ax = plt.subplots(figsize=(10, 10))
    if theta is not None:
        theta = np.asarray(theta)
        (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()
        model_output = DiagenesisModel_d18O(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2)
        ax.plot(model_output[0], tspace1, label = f'top model-theta {np.round(theta, decimals=4)}')
        ax.plot(model_output[1], tspace2, label = f'bottom model-theta {np.round(theta, decimals=4)}')


    ax.errorbar(bcd.d18O, bcd.tspace_data_d18O, xerr=bcd.d18O_err, fmt="o", color="k", label="stolper + turner + schrag data")
    ax.invert_yaxis()
    ax.set_title('d18O v. age/depth')
    ax.set_xlabel('d18O permil VSMOW')
    ax.set_ylabel('Age (Mya)')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()

def plot_D47_expR(theta = None, boundary_conditions_and_data = None):
    """
    Function will default plot the D47 data. Given a theta = [beta, gamma, alpha, model_s], it will plot the mean model output.
    """
    bcd = boundary_conditions_and_data
    fig, ax = plt.subplots(figsize=(10, 10))
    if theta is not None:
        theta = np.asarray(theta)
        (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()
        model_output = DiagenesisModel_D47_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)
        ax.plot(model_output[0], tspace1, label = f'top model-theta {np.round(theta, decimals=4)}')
        ax.plot(model_output[1], tspace2, label = f'bottom model-theta {np.round(theta, decimals=4)}')


    ax.errorbar(bcd.D47, bcd.tspace_data, xerr=bcd.D47_err, fmt="o", color="k", label="stolper + turner data")
    ax.invert_yaxis()
    ax.set_title('D47 v. age/depth')
    ax.set_xlabel('D47 permil')
    ax.set_ylabel('Age (Mya)')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()

def plot_T_clump_expR(theta = None, boundary_conditions_and_data = None):
    """
    Function will default plot the T_clump data. Given a theta = [beta, gamma, alpha, model_s], it will plot the mean model output.
    """
    bcd = boundary_conditions_and_data
    fig, ax = plt.subplots(figsize=(10, 10))
    if theta is not None:
        theta = np.asarray(theta)
        (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()
        model_output = DiagenesisModel_D47_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, geotherm1, geotherm2)
        ax.plot(anderson2T(model_output[0]), tspace1, label = f'top model-theta {np.round(theta, decimals=4)}')
        ax.plot(anderson2T(model_output[1]), tspace2, label = f'bottom model-theta {np.round(theta, decimals=4)}')


    ax.errorbar(anderson2T(bcd.D47), bcd.tspace_data, xerr=bcd.D47_err, fmt="o", color="k", label="stolper + turner data")
    ax.invert_yaxis()
    ax.set_title('T_clump v. age/depth')
    ax.set_xlabel('Temperature (deg C)')
    ax.set_ylabel('Age (Mya)')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()

def plot_d18O_expR(theta = None, boundary_conditions_and_data = None):
    """
    Function will default plot the d18O data. Given a theta = [beta, gamma, alpha], it will plot a model. 
    """
    bcd = boundary_conditions_and_data
    fig, ax = plt.subplots(figsize=(10, 10))
    if theta is not None:
        theta = np.asarray(theta)
        (tspace1, tspace2, T_deep1, T_deep2, depth1, depth2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2) = bcd.mean_all_BCs()
        model_output = DiagenesisModel_d18O_expR(theta, tspace1, tspace2, T_deep1, T_deep2, SST1, SST2, seawater_d18O1, seawater_d18O2, geotherm1, geotherm2)
        ax.plot(model_output[0], tspace1, label = f'top model-theta {np.round(theta, decimals=4)}')
        ax.plot(model_output[1], tspace2, label = f'bottom model-theta {np.round(theta, decimals=4)}')


    ax.errorbar(bcd.d18O, bcd.tspace_data_d18O, xerr=bcd.d18O_err, fmt="o", color="k", label="stolper + turner + schrag data")
    ax.invert_yaxis()
    ax.set_title('d18O v. age/depth')
    ax.set_xlabel('d18O permil VSMOW')
    ax.set_ylabel('Age (Mya)')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()

def plot_R(theta, boundary_conditions_and_data):
    """
    Plot recrystallization with time. theta = [beta, gamma, alpha, model_s]
    """
    bcd = boundary_conditions_and_data
    tspace = bcd.coarse_time.numpy()
    mask1 = tspace < bcd.t_switch*np.ones(len(tspace))
    mask2 = tspace >= bcd.t_switch*np.ones(len(tspace))
    R_top = np.zeros_like(tspace)
    for i in range(len(tspace)):
        R_top[i] = recryst(tspace[i], theta)
    R = mask1*R_top + mask2*theta[2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(R*100, tspace, label="R")
    ax.invert_yaxis()
    ax.set_title('R v. time/depth')
    ax.set_xlabel('Recrystallization %/Ma')
    ax.set_ylabel('time Mya')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()

def plot_R_expR(theta, boundary_conditions_and_data):
    """
    Plot recrystallization with time. theta = [beta, gamma, alpha, model_s]
    """
    bcd = boundary_conditions_and_data
    tspace = bcd.coarse_time.numpy()
    R = np.zeros_like(tspace)
    for i in range(len(tspace)):
        R[i]  = recryst(tspace[i], theta)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(R*100, tspace, label="R")
    ax.invert_yaxis()
    ax.set_title('R v. time/depth')
    ax.set_xlabel('Recrystallization %/Ma')
    ax.set_ylabel('time Mya')

    secax = ax.secondary_yaxis('right', functions=(age2depth, depth2age))
    secax.set_ylabel('Depth (mbsf)')

    plt.legend()
    plt.show()
