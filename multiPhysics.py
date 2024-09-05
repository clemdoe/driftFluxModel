# Version: 2021.09.22
# Author : Clément HUET, Raphaêl Guasch


from thermalHydraulics.THM import Version5_THM_prototype
from thermalHyraulics.THM import plotting
from iapws import IAPWS97
import numpy as np

# Algorithm parameters
nIter = 100
tol = 1e-4
underRelaxationFactor = 0.5

# Fonction used for the convergence algorithm

def underRelaxation(Field, OldField, underRelaxationFactor):
    return underRelaxationFactor*Field + (1-underRelaxationFactor)*OldField

def convergence(Field, OldField, tol):
    return np.abs(Field-OldField) < tol

# Fields of the problem
TeffFuel = []
Twater = []
rho = []
Qfiss = []

# Initial thermal hydraulic resolution

# MultiPhysics resolution
for i in range(nIter):
    
    ################## Nuclear part ##################

    ############# Thermalhydraulic part ##############
    case1 = Version5_THM_prototype("Testing THM Prototype", rw1, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "constant", 
    fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, 
    startHeating = 0.1, stopHeating= Lf1, voidFractionCorrel= voidFractionCorrel)

    TeffTEMP, TwaterTEMP, rhoTEMP = case1.get_nuclear_parameters()
    TeffFuel.append(TeffTEMP)
    Twater.append(TwaterTEMP)
    rho.append(rhoTEMP)

    ############## Under relaxation #################
    TeffFuel[-1] = underRelaxation(TeffFuel[-1], TeffFuel[-2], underRelaxationFactor)
    Twater[-1] = underRelaxation(Twater[-1], Twater[-2], underRelaxationFactor)
    rho[-1] = underRelaxation(rho[-1], rho[-2], underRelaxationFactor)

    ############## Convergence test #################
    if convergence(TeffFuel[-1], TeffFuel[-2], tol) and convergence(Twater[-1], Twater[-2], tol) and convergence(rho[-1], rho[-2], tol):
        print("Convergence reached after ", i, " iterations")
        break

    if i == nIter-1:
        print("Convergence not reached after ", i, " iterations")
    


    
