#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet

from THM_main import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
from THM_main import plotting
# Begining of the script used to test the THM prototype class.
compute_case_imposedPower = False
compute_case_12223_old = False
compute_case_12223 = False
compute_case_transient = True


if compute_case_transient:
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    #Geometric parameters:
    fuel_radius1 = 5.6*10**(-3) #External radius of the fuel m
    gap_width1 = 0.54e-3 # m
    clad_width1 = 0.38e-3 # m
    clad_radius = fuel_radius1 + gap_width1 + clad_width1 #External radius of the clad m
    canal_type1 = "square" #"square" #"cylindrical"
    Poro = 0.5655077285
    canal_width1 = 0.0157 * np.sqrt(Poro)
    Lf1 = 1.655 # m
    rw1=canal_width1 # canal radius
    gap_rad1 = fuel_radius1+gap_width1
    clad_rad1 = gap_rad1+clad_width1

    #Material properties:
    k_fuel1 = 5 # W/m/K
    H_gap1 = 10000 # W/m^2/K
    k_clad1 = 10 # W/m/K
    
    #Nuclear parameters:
    Qmax = 5e7 # W/m^3
    Qtype = "constant"
    
    #Meshing parameters:
    I_f1 = 8
    I_c1 = 3
    I_z1 = 10

    #Thermohydraulic parameters:
    flowArea = rw1 ** 2 - np.pi * clad_rad1 ** 2
    T_in1 = 602.75 # K
    u_inlet = 4.68292412 #m/s
    pOutlet = 14739394.95 
    pressureDrop = 786737 #Pa/m
    falsePInlet = pOutlet - Lf1 * pressureDrop
    rhoInlet = IAPWS97(T = T_in1, P = falsePInlet*10**(-6)).rho #kg/m3
    print(f'rhoInlet = {rhoInlet}, pInlet = {falsePInlet}, flowArea = {flowArea}')
    Q_flow1 = u_inlet * rhoInlet * flowArea # kg/m^2/s
    print(f'Q_flow1 = {Q_flow1}')

    #Correlation used:
    voidFractionCorrel = "HEM1"

    Qfiss1 = []
    for i in range(I_z1): 
        if Qtype == 'constant':
            Qfiss1.append(Qmax)
        elif Qtype == 'sinusoidal':
            Qfiss1.append(Qmax*np.sin(i*np.pi)) #volumetric heat generation rate W/m3
    
    case1 = Version5_THM_prototype("Testing THM Prototype", canal_type1, rw1, fuel_radius1, gap_rad1, clad_rad1, Lf1, T_in1, pOutlet, Q_flow1, Qfiss1,
                                k_fuel1, H_gap1, k_clad1, I_z1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, voidFractionCorrel= voidFractionCorrel, frfaccorel = 'blasius')

    plotter = plotting([case1])#, case2, case3])#
    plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    #plotter.GenFoamComp("BWR\driftFluxModel\hermalHydraulics\compOpenFoam.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True])

