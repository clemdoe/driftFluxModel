#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet

from THM_main import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
from THM_main import plotting
# Begining of the script used to test the THM prototype class.

compute_case_transient = False
compute_case_real = False
compute_case_genfoam_OLD_Ex1_12223  = False
compute_case_genfoam_NEW_Ex1_12223  = True
compute_case_genfoam_comparaison_correl = False
compute_case_paths = False
compute_case_multiphysics = False
compute_case_BFBT = False

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
    voidFractionCorrel = "HEM1"    #choice between 'EPRIvoidModel' and 'GEramp' and 'modBestion' and 'HEM1'
    frfaccorel = "Churchill"                #choice between 'Churchill' and 'blasius'
    P2Pcorel = "HEM2"                       #choice between 'HEM1' and 'HEM2' and 'MNmodel'
    numericalMethod = 'GaussSiedel'            #choice between 'BiCG', 'BiCGStab', 'GaussSiedel' and 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220 * ((np.pi * waterRadius**2) - (np.pi * cladRadius**2)) / (Iz1 * np.pi * fuelRadius**2) #W/m3
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
        
    case1 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    plotter = plotting([case1])#, case2, case3])#
    #plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)

if compute_case_genfoam_comparaison_correl:
    case_name = "genfoam_comparaison_correl"
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square"
    waterRadius = 0.0133409 # m
    fuelRadius = 0.00542310/2 # m : fuel rod radius
    gapRadius = 0  # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
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
    plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    #plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)

    pass

if compute_case_multiphysics:
    
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
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"
    frfaccorel = "Churchill"
    P2Pcorel = "HEM2"
    numericalMethod = "BiCG"

    ############ Nuclear Parameters ###########
    ## Fission parameters
    # specific power = 38.6 W/g
    specificPower = 38.60 # W/g, multiplied by 5 to have a more realistic value and create boiling
    PFiss = specificPower*Fuel_mass*1000 # W
    print("PFiss = ", PFiss)
    qFiss_init = guessAxialPowerShape(PFiss, Iz1, height, fuelRadius)

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
    

    plotter = plotting([THComponent])#, case2, case3, case4])
    plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])


if compute_case_paths:

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
    height = 4.5 # m : height : active core height in BWR/6
    

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
    Iz1 = 20 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "HEM1"
    frfaccorel = "base"
    P2Pcorel = "base"
    numericalMethod = "FVM"

    ############ Nuclear Parameters ###########
    ## Fission parameters
    # specific power = 38.6 W/g
    specificPower = 38.6 # W/g
    PFiss = specificPower*Fuel_mass*1000*100 # W

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

    ########## Fields of the TH problem ##########
    TeffFuel = []
    Twater = []
    rho = []
    Qfiss = []

    qFiss = guessAxialPowerShape(PFiss, Iz1, height)
    print(f"qFiss = {qFiss}")
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