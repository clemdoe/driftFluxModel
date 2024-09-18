#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet

from THM_main import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
from THM_main import plotting
# Begining of the script used to test the THM prototype class.

compute_case_transient = False
compute_case_real = True


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
    pOutlet = 14739394.95 #Pa
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

if compute_case_real:
    case_name = "Atrium10"
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Fluid parameters
    # T_inlet, T_outlet = 270, 287 Celcius
    # Nominal coolant flow rate = 1530 kg/s
    # Nominal operating pressure = 7.2 MPa (abs)
    #hInlet =  # to fill
    pOutlet =  7.2e6 # Pa
    tInlet = 270 + 273.25
    #qFlow =  # to fill
    massFlowRate = 1530/(91*200) # kg/s

    ## Geometric parameters
    canalType = "square"
    waterRadius = 1.295e-2 # m ATRIUM10 pincell pitch
    fuelRadius = 0.4435e-2 # m : fuel rod radius
    gapRadius = 0.4520e-2 # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius = 0.5140e-2 # m : clad external radius
    height = 3.8 # m : height : active core height in BWRX-300 SMR

    ## Additional parameters needed for the calculation
    volumic_mass_UOX = 10970 # kg/m3
    Fuel_volume = np.pi*fuelRadius**2*height # m3
    Fuel_mass = Fuel_volume*volumic_mass_UOX # kg

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 10 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "HEM1"
    frfaccorel = "base"
    P2Pcorel = "base"

    ############ Nuclear Parameters ###########
    ## Fission parameters
    # specific power = 38.6 W/g
    specificPower = 38.6 # W/g
    PFiss = specificPower*Fuel_mass*1000 # W

    qFiss = PFiss/Fuel_volume # W/m3

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000 
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4
    underRelaxationFactor = 0.5

    #Correlation used:
    voidFractionCorrel = "HEM1"

    Qfiss1 = []
    for i in range(Iz1): 
        Qfiss1.append(qFiss)
        
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, massFlowRate, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel)
    
    
    plotter = plotting([case1])#, case2, case3])#
    plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    #plotter.GenFoamComp("BWR\driftFluxModel\hermalHydraulics\compOpenFoam.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True])

