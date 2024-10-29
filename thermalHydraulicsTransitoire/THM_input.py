#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet

from THM_main import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
from THM_main import plotting
import pandas as pd
import matplotlib.pyplot as plt
# Begining of the script used to test the THM prototype class.

compute_case_transient = False
compute_case_real = False
compute_case_genfoam_OLD_Ex1_12223  = False
compute_case_genfoam_NEW_Ex1_12223  = True
compute_case_genfoam_comparaison_numericalMethod = False
compute_case_genfoam_comparaison_P2Pcorel = False
compute_case_genfoam_comparaison_frfaccorel = False
compute_case_genfoam_comparaison_voidFractionCorrel = False
compute_case_paths = False
compute_case_multiphysics = False
compute_case_BFBT = False
compute_case_compare_power = False
compute_case_compare_Dh = False
compute_case_compare_Dz = False


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
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "blasius"
    P2Pcorel = "HEM1"

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

if compute_case_genfoam_OLD_Ex1_12223:
    case_name = "PSBT BenchMark Ex1 12223"
    #User choice:
    solveConduction = True
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.0133409 # m
    fuelRadius = 0.00542310/2 # m : fuel rod radius
    gapRadius = 0.00542310/2 + 0.00001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius =  0.0094996/2 # m : clad external radius
    height = 1.655 # m : height : active core height in BWRX-300 SMR

    ## Fluid parameters
    pOutlet = 14719781.65 # Pa
    tInlet = 602.75 #K
    u_inlet = 4.467092221 #m/s
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
    flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
    qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "blasius"
    P2Pcorel = "HEM1"
    numericalMethod = 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220 # W/m3   ##############FALSE NEED TO CHANGE TO FIT WITH THE OLD

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4

    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
        
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    plotter = plotting([case1])#, case2, case3])#
    #plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    plotter.GenFoamComp(r"C:\Users\sapaq\Downloads\Clem tempo\driftFluxModel\thermalHydraulicsTransitoire\Firstopenfoam.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True])

if compute_case_genfoam_NEW_Ex1_12223:
    case_name = "PSBT BenchMark Ex1 12223"
    #User choice:
    solveConduction = False
    plot_at_z1 = []

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.0133409 # m
    fuelRadius = 0.00542310/2 # m : fuel rod radius
    gapRadius = fuelRadius + 0.0000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius =  0.0094996/2 # m : clad external radius
    height = 1.655 # m : height : active core height in BWRX-300 SMR

    ## Fluid parameters
    pOutlet = 14719781.65 # Pa
    tInlet = 592.75 #K
    u_inlet = 4.467092221 #m/s
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
    flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
    qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"    #choice between 'EPRIvoidModel' and 'GEramp' and 'modBestion' and 'HEM1'
    frfaccorel = "base"                #choice between 'Churchill' and 'blasius'
    P2Pcorel = "base"                       #choice between 'HEM1' and 'HEM2' and 'MNmodel'
    numericalMethod = 'FVM'            #choice between 'BiCG', 'BiCGStab', 'GaussSiedel' and 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220 #W/m3
    #qFiss = 1943301220 # W/m3
    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4

    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)

        
    frfaccorel = "Churchill"
    case2 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    frfaccorel = "blasius"
    case3 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)


    plotter = plotting([case2, case3])#
    #plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'frfaccorel', [True, True, True, True, True, True], genFoamVolumeFraction)

if compute_case_genfoam_comparaison_numericalMethod:
    case_name = "genfoam_comparaison_correl"
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.0133409 # m
    fuelRadius = 0.00542310/2 # m : fuel rod radius
    gapRadius = fuelRadius + 0.0000000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius =  0.0094996/2 # m : clad external radius
    height = 1.655 # m : height : active core height in BWRX-300 SMR

    ## Fluid parameters
    pOutlet = 14719781.65 # Pa
    tInlet = 592.75 #K
    u_inlet = 4.467092221 #m/s
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
    flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
    qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "Churchill"
    P2Pcorel = "HEM2"
    numericalMethod = 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220 # W/m3

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4

    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
        
    numericalMethod = 'FVM'
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    numericalMethod = 'BiCGStab'
    case2 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    numericalMethod = 'GaussSiedel'
    case3 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    numericalMethod = 'BiCG'
    case4 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    plotter = plotting([case1, case2, case3, case4])
    #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'numericalMethod', [True, True, True, True, True, True], genFoamVolumeFraction)
    plotter.writeResults(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\resultsDFM.xlsx")
    plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "numericalMethod", genFoamVolumeFraction)

if compute_case_genfoam_comparaison_frfaccorel:
    case_name = "genfoam_comparaison_correl"
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.0133409 # m
    fuelRadius = 0.00542310/2 # m : fuel rod radius
    gapRadius = fuelRadius + 0.0000000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius =  0.0094996/2 # m : clad external radius
    height = 1.655 # m : height : active core height in BWRX-300 SMR

    ## Fluid parameters
    pOutlet = 14719781.65 # Pa
    tInlet = 592.75 #K
    u_inlet = 4.467092221 #m/s
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
    flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
    qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "Churchill"
    P2Pcorel = "HEM2"
    numericalMethod = 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220 # W/m3

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4

    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
        
    frfaccorel = "Churchill"
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    frfaccorel = "blasius"
    case2 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    plotter = plotting([case1, case2])
    #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'frfaccorel', [True, True, True, True, True, True], genFoamVolumeFraction)
    plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "frfaccorel", genFoamVolumeFraction)
    plotter.writeResults(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\resultsDFM.xlsx")

if compute_case_genfoam_comparaison_P2Pcorel:
    case_name = "genfoam_comparaison_correl"
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.0133409 # m
    fuelRadius = 0.00542310/2 # m : fuel rod radius
    gapRadius = fuelRadius + 0.0000000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius =  0.0094996/2 # m : clad external radius
    height = 1.655 # m : height : active core height in BWRX-300 SMR

    ## Fluid parameters
    pOutlet = 14719781.65 # Pa
    tInlet = 592.75 #K
    u_inlet = 4.467092221 #m/s
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
    flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
    qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "Churchill"
    P2Pcorel = "HEM2"
    numericalMethod = 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220 # W/m3

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4

    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
        
    P2Pcorel = "HEM1"
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    P2Pcorel = "HEM2"
    case2 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    P2Pcorel = "MNmodel"
    case3 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)


    plotter = plotting([case1, case2, case3])
    #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'P2Pcorrel', [True, True, True, True, True, True], genFoamVolumeFraction)
    plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "P2Pcorrel", genFoamVolumeFraction)
    plotter.writeResults(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\resultsDFM.xlsx")

if compute_case_genfoam_comparaison_voidFractionCorrel:
    case_name = "genfoam_comparaison_correl"
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.0133409 # m
    fuelRadius = 0.00542310/2 # m : fuel rod radius
    gapRadius = fuelRadius + 0.0000000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius =  0.0094996/2 # m : clad external radius
    height = 1.655 # m : height : active core height in BWRX-300 SMR

    ## Fluid parameters
    pOutlet = 14719781.65 # Pa
    tInlet = 592.75 #K
    u_inlet = 4.467092221 #m/s
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
    flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
    qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "Churchill"
    P2Pcorel = "MNmodel"
    numericalMethod = 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220 # W/m3

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4

    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
        
    voidFractionCorrel= "EPRIvoidModel"
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    voidFractionCorrel= "GEramp"
    case2 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    voidFractionCorrel= "modBestion"
    case3 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    voidFractionCorrel= "HEM1"
    case4 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    plotter = plotting([case1, case2, case3, case4])
    #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)
    plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "voidFractionCorrel", genFoamVolumeFraction)

    plotter.writeResults(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\resultsDFM.xlsx")

if compute_case_multiphysics:
    
    def guessAxialPowerShape(Ptot, Iz, height, radius):
        volume = np.pi * radius**2 * height
        heights = np.linspace(0, height, Iz)

        power_profile = lambda h: np.sin(np.pi * h / height) * ((Ptot*np.pi)/(2*volume))
        
        volumic_powers = []
        
        for i in range(Iz):
            h = heights[i]
            print(f"Height = {h}")
            
            power_density = power_profile(h)
            print(f"Power density = {power_density}")
            volumic_powers.append(power_density)
        
        return volumic_powers

    ######## End functions declaration ##########


    ########## User input ##########

    solveConduction = True
    zPlotting = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 1.295e-2 # m ATRIUM10 pincell pitch
    fuelRadius = 0.4435e-2 # m : fuel rod radius
    gapRadius = 0.4520e-2 # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius = 0.5140e-2 # m : clad external radius
    height = 3.8 # m : height : active core height in BWRX-300 SMR


    ## Fluid parameters
    tInlet = 270 + 273.15 # K
    # T_inlet, T_outlet = 270, 287 Celcius
    pOutlet =  7.2e6 # Pa
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
    flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2

    # Nominal coolant flow rate = 1530 kg/s
    # Nominal operating pressure = 7.2 MPa (abs)
    massFlowRate = 1530  / (200*91)  # kg/s

    ## Additional parameters needed for the calculation
    solveConduction = True
    volumic_mass_U = 19000 # kg/m3
    Fuel_volume = np.pi*fuelRadius**2*height # m3
    Fuel_mass = Fuel_volume*volumic_mass_U # kg

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 75 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "Churchill"
    P2Pcorel = "HEM2"
    numericalMethod = "BiCG"

    ############ Nuclear Parameters ###########
    ## Fission parameters
    # specific power = 38.6 W/g
    specificPower = 18.60 # W/g, multiplied by 5 to have a more realistic value and create boiling
    PFiss = specificPower*Fuel_mass*1000 # W
    print("PFiss = ", PFiss)
    qFiss= guessAxialPowerShape(PFiss, Iz1-5, height, fuelRadius)
    print("qFiss_init = ", qFiss)
    qFiss_init = [0,0,0,0,0] + list(qFiss)
    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000 
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4
    underRelaxationFactor = 0.5

    ########## Fields of the TH problem ##########
    TeffFuel = []
    Twater = []
    rho = []
    Qfiss = []

    ## Initial thermal hydraulic resolution
    THComponent = Version5_THM_prototype("Testing THM Prototype", canalType, waterRadius, fuelRadius, gapRadius, cladRadius, 
                                height, tInlet, pOutlet, massFlowRate, qFiss_init, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting, 
                                solveConduction, dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, 
                                numericalMethod = numericalMethod)
    
    print(THComponent.convection_sol.U[-1])

    plotter = plotting([THComponent])#, case2, case3, case4])
    plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])

if compute_case_paths:

    def guessAxialPowerShape(Ptot, Iz, height, radius):
        """
        Ptot : float : total power released (W)
        Iz : int : number of control volumes in the axial direction
        height : float : height of the fuel rod (m)
        radius : float : radius of the fuel rod (m)
        return : np.array : axial power shape with a sine shape units (W/m3)
                            --> corresponds to the power density in each control volume 
                            !! Issue with IAPWS tables when dividing by Iz
        """
        volume = np.pi * radius**2 * height
        
        # Heights of each control volume (equally spaced along the tube height)
        heights = np.linspace(0, height, Iz + 1)
        
        # Define the power profile as a sine function of height
        power_profile = lambda h: np.sin(np.pi * h / height)
        
        # Compute the volumic power for each control volume
        volumic_powers = []
        total_integral = 0
        
        for i in range(Iz):
            # Midpoint of the control volume
            h_mid = 0.5 * (heights[i] + heights[i + 1])
            print(f"Height = {h_mid}")
            
            # Power density at this control volume
            power_density = power_profile(h_mid)
            print(f"Power density = {power_density}")
            
            # Volume of this control volume
            dz = (heights[i + 1] - heights[i])
            
            # Store the volumic power (W/m^3)
            volumic_powers.append(power_density)
            
            # Update total integral for normalization
            total_integral += power_density * dz

        print(f"Total_integral = {total_integral}")
        
        # Normalize the volumetric powers so the total power matches Ptot
        volumic_powers = np.array(volumic_powers) * Ptot /(total_integral*np.pi*radius**2)/Iz
        print(f"Volumic powers = {volumic_powers}")
        total_power = np.sum(volumic_powers) * volume
        print(f"Total power = {total_power}")
        
        return volumic_powers   

    ########## User input ##########

    solveConduction = False
    zPlotting = []

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.102064 #1.295e-2 # m ATRIUM10 pincell pitch
    fuelRadius = 0.4435e-2 # m : fuel rod radius
    gapRadius = 0.4520e-2 # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius = 0.5140e-2 # m : clad external radius
    height = 4.5 # m : height : active core height in BWR/6
    

    ## Fluid parameters
    hInlet = 1220 # kJ/kg
    #tInlet = 270 + 273.15 # K
    # T_inlet, T_outlet = 270, 287 Celcius
    pOutlet =  7.25e6 # Pa
    falsePInlet = 7.4e6 # Pa
    tInlet = IAPWS97(P = falsePInlet*10**(-6), h = hInlet).T #K

    # Nominal coolant flow rate = 1530 kg/s
    # Nominal operating pressure = 7.2 MPa (abs)
    massFlowRate = 8.45  # kg/s

    ## Additional parameters needed for the calculation
    solveConduction = False

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "blasius"
    P2Pcorel = "MNmodel"
    numericalMethod = "FVM"

    """############ Nuclear Parameters ###########
    ## Fission parameters
    # specific power = 38.6 W/g
    specificPower = 38.6 # W/g
    PFiss = specificPower*Fuel_mass*1000*100 # W
    qFiss = PFiss/Fuel_volume # W/m3
    qFiss = guessAxialPowerShape(PFiss, Iz1, height)
    print(f"qFiss = {qFiss}") """

    PFiss = 2.5e6*Iz1 # W
    qFiss = guessAxialPowerShape(PFiss, Iz1, height, cladRadius)
    print(f"qFiss = {qFiss}")

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000 
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    
    ########## Fields of the TH problem ##########
    TeffFuel = []
    Twater = []
    rho = []
    Qfiss = []

    ## Initial thermal hydraulic resolution
    THComponent = Version5_THM_prototype("Testing THM Prototype", canalType, waterRadius, fuelRadius, gapRadius, cladRadius, 
                                height, tInlet, pOutlet, massFlowRate, qFiss, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting, 
                                solveConduction, dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, 
                                numericalMethod = numericalMethod)
    

    plotter = plotting([THComponent])#, case2, case3, case4])
    plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])

if compute_case_BFBT:
    case_name = "BFBT"
    #User choice:
    solveConduction = True
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 104.7801951 *(10**(-3))# m
    fuelRadius = 13.01793635 *(10**(-3))# m : fuel rod radius
    gapRadius = fuelRadius + 0.0000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius =  19.52690452*(10**(-3)) # m : clad external radius
    height = 3.708 # m : height : active core height in BWRX-300 SMR

    ## Fluid parameters
    pOutlet = 966000 # Pa
    hInlet =  45.1 #kJ/kg
    #tInlet = 592.75 #K
    pressureDrop = 186737 #Pa/m
    falsePInlet = pOutlet - height * pressureDrop
    print(f"falsePInlet = {falsePInlet}")
    print(f'hInlet = {hInlet}')
    tInlet = IAPWS97(P = falsePInlet*10**(-6), h = hInlet).T #K
    flowArea = 9781.5e-6 # m^2
    qFlow = 10.16*1000/3600 # kg/s

    ## Meshing parameters:
    If = 8
    I1 = 3
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "Churchill"
    P2Pcorel = "HEM2"
    numericalMethod = 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    volume = np.pi * fuelRadius**2
    qFiss = 820000 / volume # W/m3
    #qFiss = 1943301220 # W/m3
    print(f"qFiss = {qFiss}")


    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
    ########## Algorithm parameters ###########
    nIter = 1000
    tol = 1e-4

    Qfiss1 = []
    for i in range(Iz1): 
            Qfiss1.append(qFiss)
    print(Qfiss1)
        
    numericalMethod = 'FVM'
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    print(f'Quality: {case1.convection_sol.xTh[-1]}')
    plotter = plotting([case1])
    plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    #plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)


if compute_case_compare_power:
    QList = [50e8]
    for j in range(len(QList)):
        case_name = "genfoam_comparaison_correl"
        #User choice:
        solveConduction = False
        plot_at_z1 = [0.8]

        ########## Thermal hydraulics parameters ##########
        ## Geometric parameters
        canalType = "square"
        waterRadius = 0.0133409 # m
        fuelRadius = 0.00542310/2 # m : fuel rod radius
        gapRadius = fuelRadius + 0.0000000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
        cladRadius =  0.0094996/2 # m : clad external radius
        height = 1.655 # m : height : active core height in BWRX-300 SMR

        ## Fluid parameters
        pOutlet = 14719781.65 # Pa
        tInlet = 592.75 #K
        u_inlet = 4.467092221 #m/s
        pressureDrop = 186737 #Pa/m
        falsePInlet = pOutlet - height * pressureDrop
        rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
        flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
        qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

        ## Meshing parameters:
        If = 8
        I1 = 3
        Iz1 = 20 # number of control volumes in the axial direction

        ## Thermalhydraulics correlation
        voidFractionCorrel = "EPRIvoidModel"
        frfaccorel = "Churchill"
        P2Pcorel = "MNmodel"
        numericalMethod = 'FVM'

        ############ Nuclear Parameters ###########
        ## Fission parameters
        qFiss = QList[j] # W/m3

        ## Material parameters
        kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
        Hgap = 10000
        kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
        # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
        ########## Algorithm parameters ###########
        nIter = 1000
        tol = 1e-4

        Qfiss1 = []
        for i in range(Iz1): 
            if i*(height/Iz1) < 0.1:
                Qfiss1.append(0)
            else:
                Qfiss1.append(qFiss)
        print(Qfiss1)
            
        voidFractionCorrel= "EPRIvoidModel"
        case1 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "GEramp"
        case2 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "modBestion"
        case3 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "HEM1"
        case4 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        plotter = plotting([case1, case2, case3, case4])
        #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
        genFoamVolumeFraction = 0.5655077285
        #plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)
        #plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "voidFractionCorrel", genFoamVolumeFraction)
        
        plotter.writeResults(rf"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\resultsDFM_q_{QList[j]}.xlsx")


if compute_case_compare_Dh:
    rw_list = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020]
    D_h = []
    dict_results ={
    }
    for j in range(len(rw_list)):
        case_name = "genfoam_comparaison_Dh"
        #User choice:
        solveConduction = False
        plot_at_z1 = [0.8]

        ########## Thermal hydraulics parameters ##########
        ## Geometric parameters
        canalType = "square"
        waterRadius = rw_list[j] # m  0.0133409 # m #cote or radius if cylindrical
        fuelRadius = 0.00542310/2 # m : fuel rod radius
        gapRadius = fuelRadius + 0.0000000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
        cladRadius =  0.0094996/2 # m : clad external radius
        height = 1.655 # m : height : active core height in BWRX-300 SMR

        ## Fluid parameters
        pOutlet = 14719781.65 # Pa
        tInlet = 592.75 #K
        u_inlet = 4.467092221 #m/s
        pressureDrop = 186737 #Pa/m
        falsePInlet = pOutlet - height * pressureDrop
        rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
        flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
        qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

        ## Meshing parameters:
        If = 8
        I1 = 3
        Iz1 = 20 # number of control volumes in the axial direction

        ## Thermalhydraulics correlation
        voidFractionCorrel = "EPRIvoidModel"
        frfaccorel = "Churchill"
        P2Pcorel = "MNmodel"
        numericalMethod = 'FVM'

        ############ Nuclear Parameters ###########
        ## Fission parameters
        qFiss = 1943301220 # W/m3

        ## Material parameters
        kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
        Hgap = 10000
        kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
        # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
        ########## Algorithm parameters ###########
        nIter = 1000
        tol = 1e-4

        Qfiss1 = []
        for i in range(Iz1): 
            if i*(height/Iz1) < 0.1:
                Qfiss1.append(0)
            else:
                Qfiss1.append(qFiss)
        print(Qfiss1)
            
        voidFractionCorrel= "EPRIvoidModel"
        case1 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
        D_h.append(case1.convection_sol.D_h)

        voidFractionCorrel= "GEramp"
        case2 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "modBestion"
        case3 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "HEM1"
        case4 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
        
        plotter = plotting([case1, case2, case3, case4])
        #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
        genFoamVolumeFraction = 0.5655077285
        #plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)
        #plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "voidFractionCorrel", genFoamVolumeFraction)
    
        dict_results[case1.convection_sol.D_h] = [case1.convection_sol.voidFraction[-1], case2.convection_sol.voidFraction[-1], case3.convection_sol.voidFraction[-1], case4.convection_sol.voidFraction[-1]]
    print(f'D_h: {{{D_h}}}')
    xDFM = case1.convection_sol.z_mesh

    # Read the Excel file
    df = pd.read_excel(rf'C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\Results\GenFoamComp_Ex1_variation\variation_D_h\GFresults.xlsx')

    # Create empty lists for each column
    columns = df.columns.tolist()
    data = [[] for _ in columns]

    # Iterate over each row and append values to the corresponding list
    for index, row in df.iterrows():
        for i, col in enumerate(columns):
            data[i].append(row[col])

    xGenFoam = data[0]
    ecart0=[]
    ecart1=[]
    ecart2=[]
    ecart3=[]
    for i in range(len(D_h)):
        

        # Créer un DataFrame pour la première série
        df1 = pd.DataFrame({'x': xGenFoam, 'y': data[i]})
        
        # Créer un DataFrame pour la deuxième série
        df2 = pd.DataFrame({'x': xDFM, 'y': dict_results[D_h[i]][0]})
        df3 = pd.DataFrame({'x': xDFM, 'y': dict_results[D_h[i]][1]})
        df4 = pd.DataFrame({'x': xDFM, 'y': dict_results[D_h[i]][2]})
        df5 = pd.DataFrame({'x': xDFM, 'y': dict_results[D_h[i]][3]})

        # Interpolation de la deuxième série sur les abscisses de la première série
        df2_interp = pd.DataFrame({'x': df1['x']})
        df2_interp['y_interp'] = np.interp(df1['x'], df2['x'], df2['y'])

        df3_interp = pd.DataFrame({'x': df1['x']})
        df3_interp['y_interp'] = np.interp(df1['x'], df3['x'], df3['y'])

        df4_interp = pd.DataFrame({'x': df1['x']})
        df4_interp['y_interp'] = np.interp(df1['x'], df4['x'], df4['y'])

        df5_interp = pd.DataFrame({'x': df1['x']})
        df5_interp['y_interp'] = np.interp(df1['x'], df5['x'], df5['y'])

        # Calcul de la différence entre les deux séries
        ecart0.append(np.linalg.norm(df1['y'] - df2_interp['y_interp']))
        ecart1.append(np.linalg.norm(df1['y'] - df3_interp['y_interp']))
        ecart2.append(np.linalg.norm(df1['y'] - df4_interp['y_interp']))
        ecart3.append(np.linalg.norm(df1['y'] - df5_interp['y_interp']))

    # Affichage de la différence
    fig, ax = plt.subplots()
    ax.plot(D_h, ecart0, label=f"Ecart {case1.convection_sol.voidFractionCorrel}.")
    ax.plot(D_h, ecart1, label=f"Ecart {case2.convection_sol.voidFractionCorrel}.")
    ax.plot(D_h, ecart2, label=f"Ecart {case3.convection_sol.voidFractionCorrel}.")
    ax.plot(D_h, ecart3, label=f"Ecart {case4.convection_sol.voidFractionCorrel}.")
    ax.set_xlabel("x")
    ax.set_ylabel("Ecart")
    ax.set_title("Ecart avec GenFoam")
    ax.legend(loc="best")
    plt.show()


if compute_case_compare_Dz:
    IzList = [10, 20, 40, 80, 160, 320, 640, 1280]
    dict_results ={
    }
    for j in range(len(IzList)):
        case_name = "genfoam_comparaison_Dz"
        #User choice:
        solveConduction = False
        plot_at_z1 = [0.8]

        ########## Thermal hydraulics parameters ##########
        ## Geometric parameters
        canalType = "square"
        waterRadius = 0.0133409 # m #cote or radius if cylindrical
        fuelRadius = 0.00542310/2 # m : fuel rod radius
        gapRadius = fuelRadius + 0.0000000001  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
        cladRadius =  0.0094996/2 # m : clad external radius
        height = 1.655 # m : height : active core height in BWRX-300 SMR

        ## Fluid parameters
        pOutlet = 14719781.65 # Pa
        tInlet = 592.75 #K
        u_inlet = 4.467092221 #m/s
        pressureDrop = 186737 #Pa/m
        falsePInlet = pOutlet - height * pressureDrop
        rhoInlet = IAPWS97(T = tInlet, P = falsePInlet*10**(-6)).rho #kg/m3
        flowArea = waterRadius ** 2 - np.pi * cladRadius ** 2
        qFlow = u_inlet * rhoInlet * flowArea # kg/m^2/s

        ## Meshing parameters:
        If = 8
        I1 = 3
        Iz1 = IzList[j] # number of control volumes in the axial direction

        ## Thermalhydraulics correlation
        voidFractionCorrel = "EPRIvoidModel"
        frfaccorel = "Churchill"
        P2Pcorel = "MNmodel"
        numericalMethod = 'FVM'

        ############ Nuclear Parameters ###########
        ## Fission parameters
        qFiss = 1943301220 # W/m3

        ## Material parameters
        kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
        Hgap = 10000
        kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
        # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity
        ########## Algorithm parameters ###########
        nIter = 1000
        tol = 1e-4

        Qfiss1 = []
        for i in range(Iz1): 
            if i*(height/Iz1) < 0.1:
                Qfiss1.append(0)
            else:
                Qfiss1.append(qFiss)
        print(Qfiss1)
            
        voidFractionCorrel= "EPRIvoidModel"
        case1 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "GEramp"
        case2 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "modBestion"
        case3 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

        voidFractionCorrel= "HEM1"
        case4 = Version5_THM_prototype(case_name, canalType,
                    waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                    kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                    dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
        
        plotter = plotting([case1, case2, case3, case4])
        #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
        genFoamVolumeFraction = 0.5655077285
        #plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)
        #plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "voidFractionCorrel", genFoamVolumeFraction)
    
        dict_results[IzList[j]] = [case1.convection_sol.voidFraction[-1], case2.convection_sol.voidFraction[-1], case3.convection_sol.voidFraction[-1], case4.convection_sol.voidFraction[-1]]

    xDFM = case1.convection_sol.z_mesh

    # Read the Excel file
    df = pd.read_excel(rf'C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\Results\GenFoamComp_Ex1_variation\variation_Iz\GFresults.xlsx')

    # Create empty lists for each column
    columns = df.columns.tolist()
    data = [[] for _ in columns]

    # Iterate over each row and append values to the corresponding list
    for index, row in df.iterrows():
        for i, col in enumerate(columns):
            data[i].append(row[col])

    xGenFoam = data[0]
    ecart0=[]
    ecart1=[]
    ecart2=[]
    ecart3=[]

    from scipy.interpolate import interp1d
    # Fonction pour interpoler la solution de référence sur les hauteurs simulées
    def interpolate_and_compare(hauteur_ref, fraction_vide_ref, hauteur_sim):
        # Interpolation de la solution de référence
        interpolation = interp1d(hauteur_ref, fraction_vide_ref, fill_value="extrapolate")
        
        # Calcul de la fraction de vide interpolée sur les hauteurs simulées
        fraction_vide_interpolee = interpolation(hauteur_sim)
        
        return fraction_vide_interpolee

    for i in range(len(IzList)):

        # Interpolation pour la première simulation
        fraction_vide_interpolée_sim = interpolate_and_compare(xGenFoam, data[1], xDFM[i])
    
        # Calcul de l'écart entre la solution de référence et la solution interpolée
        ecart0.append(np.mean(abs(dict_results[IzList[i]][0] - fraction_vide_interpolée_sim)))
        ecart1.append(np.mean(abs(dict_results[IzList[i]][1] - fraction_vide_interpolée_sim)))
        ecart2.append(np.mean(abs(dict_results[IzList[i]][2] - fraction_vide_interpolée_sim)))
        ecart3.append(np.mean(abs(dict_results[IzList[i]][3] - fraction_vide_interpolée_sim)))

    # Affichage de la différence
    fig, ax = plt.subplots()
    ax.plot(IzList, ecart0, label=f"Ecart {case1.convection_sol.voidFractionCorrel}.")
    ax.plot(IzList, ecart1, label=f"Ecart {case2.convection_sol.voidFractionCorrel}.")
    ax.plot(IzList, ecart2, label=f"Ecart {case3.convection_sol.voidFractionCorrel}.")
    ax.plot(IzList, ecart3, label=f"Ecart {case4.convection_sol.voidFractionCorrel}.")
    ax.set_xlabel("Raffinement en hauteur")
    ax.set_ylabel("Ecart")
    ax.set_title("Ecart avec GenFoam")
    ax.legend(loc="best")
    plt.show()
