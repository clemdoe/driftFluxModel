#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet, Raphael Guasch


from THM import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
from THM import plotting
# Begining of the script used to test the THM prototype class.
compute_case_imposedPower = True
compute_case_12223_old = False
compute_case_12223 = False


#self.T_in = T_inlet # inlet water temperature K
        #self.P_cool = P_inlet # coolant pressure in MPa, assumed to be constant along the axial profile.
        #IAPWS97(T = self.T_in, P = self.P_cool).h * 1000 #J/kg
        #self.pOutlet =  14739394.95 #Pa


if compute_case_imposedPower:
    #User choice:
    solveConduction = True
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
    Qmax = 5e8 # W/m^3
    Qtype = "constant"
    #Meshing parameters:
    I_f1 = 8
    I_c1 = 3
    I_z1 = 10

    #Thermohydraulic parameters:
    T_in1 = 602.75 # K
    u_inlet = 4.68292412 #m/s
    Q_flow1 = u_inlet * 1000 # kg/m^2/s
    P_inlet = 14.9 * (10**6) #Pa
    hInlet = IAPWS97(T = T_in1, P = P_inlet * 10**(-6)).h * 1000 #J/kg
    pOutlet = 14739394.95 

    #Correlation used:
    voidFractionCorrel = "HEM1"

    Qfiss1 = []
    for i in range(I_z1): 
        if Qtype == 'constant':
            Qfiss1.append(Qmax)
        elif Qtype == 'sinusoidal':
            Qfiss1.append(Qmax*np.sin(i*np.pi)) #volumetric heat generation rate W/m3
    
    case1 = Version5_THM_prototype("Testing THM Prototype", canal_type1, rw1, fuel_radius1, gap_rad1, clad_rad1, Lf1, hInlet, pOutlet, Q_flow1, Qfiss1,
                                k_fuel1, H_gap1, k_clad1, I_z1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, voidFractionCorrel= voidFractionCorrel)
    #case2 = Version5_THM_prototype("Testing THM Prototype", rw1, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "constant", 
                                #fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, voidFractionCorrel="modBestion")
    #case3 = Version5_THM_prototype("Testing THM Prototype", rw1, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "constant",
                                #fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, voidFractionCorrel="HEM1")
    #case4 = Version5_THM_prototype("Testing THM Prototype", rw1, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "constant",
                                #fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, voidFractionCorrel="GEramp")
    print(f"case 1 h_z is {case1.convection_sol.h_z} J/kg")
    print(f"case 1 T_water is {case1.convection_sol.T_water} K")
    print(f"case 1 Hc is {0.5*(case1.convection_sol.Hc[3]+case1.convection_sol.Hc[4])} W/m^2/K")
    print(f"q_fluid1 = {case1.convection_sol.q__}")
    
    print(f"case 1 A_canal = {case1.convection_sol.flowArea} m^2")
    print(f"case 1 T_surf is {case1.convection_sol.T_surf} K")

    if solveConduction:
        print(f"case 1 T_eff in fuel is {case1.T_eff_fuel} K")
        print(f"case 1 T_surf fuel is {case1.T_fuel_surface} K")
    #case1.compare_with_THM_DONJON("C:/Users/cleme/OneDrive/Documents/Poly/BWR/driftFluxModel/THM_prototypeDFM/pincell_mphy_thm_devoir.result",[True, True, True, True, True, True])
    #case1.plot_Temperature_at_z(1)
    case1.get_nuclear_parameters()
    if solveConduction:
        case1.plotColorMap()

    #case1.plotThermohydraulicParameters([True, True, True, True, True, True])
    
    plotter = plotting([case1])#, case2, case3, case4])
    #plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    plotter.GenFoamComp("BWR\driftFluxModel\THM_prototypeV1\Firstopenfoam.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True])


if compute_case_12223_old:
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    #Geometric parameters:
    fuel_radius1 = 0.8*10**(-3) #External radius of the fuel m
    gap_width1 = 0 # m
    clad_width1 = 0 # m
    clad_radius = fuel_radius1 + gap_width1 + clad_width1 #External radius of the clad m
    canal_type1 = "cylindrical" #"square" #"cylindrical"
    Poro = 0.5655077285
    canal_width1 = 4.75*10**(-3) * np.sqrt(Poro)
    Lf1 = 1.655 # m
    rw1=canal_width1 # canal radius
    gap_rad1 = fuel_radius1+gap_width1
    clad_rad1 = gap_rad1+clad_width1

    #Material properties:
    k_fuel1 = 5 # W/m/K
    H_gap1 = 10000 # W/m^2/K
    k_clad1 = 10 # W/m/K
    
    #Nuclear parameters:
    Qfiss1 = 1943301220 / 0.5655077285 # W/m^3
    
    #Meshing parameters:
    I_f1 = 8
    I_c1 = 3
    I_z1 = 10

    #Thermohydraulic parameters:
    T_in1 = 592.75 # K
    u_inlet = 4.467092221 #m/s
    Q_flow1 = u_inlet * 1000 # kg/m^2/s
    P_inlet = 14744900.0 #Pa
    hInlet = IAPWS97(T = T_in1, P = P_inlet * 10**(-6)).h * 1000 #J/kg
    pOutlet = 14719781.65

    #Correlation used:
    voidFractionCorrel = "HEM1"
    
    case1 = Version5_THM_prototype("Testing THM Prototype", rw1, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "constant", 
                                fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, startHeating = 0.1, stopHeating= Lf1, frfaccorel = 'base', voidFractionCorrel = voidFractionCorrel)

    """ print(f"case 1 h_z is {case1.convection_sol.h_z} J/kg")
    print(f"case 1 T_water is {case1.convection_sol.T_water} K")
    print(f"case 1 Hc is {0.5*(case1.convection_sol.Hc[3]+case1.convection_sol.Hc[4])} W/m^2/K")
    print(f"q_fluid1 = {case1.convection_sol.q__}")
    
    print(f"case 1 A_canal = {case1.convection_sol.flowArea} m^2")
    print(f"case 1 T_surf is {case1.convection_sol.T_surf} K") """

    #if solveConduction:
    #    print(f"case 1 T_eff in fuel is {case1.T_eff_fuel} K")
    #    print(f"case 1 T_surf fuel is {case1.T_fuel_surface} K")
    #case1.compare_with_THM_DONJON("C:/Users/cleme/OneDrive/Documents/Poly/BWR/driftFluxModel/THM_prototypeDFM/pincell_mphy_thm_devoir.result",[True, True, True, True, True, True])
    #case1.plot_Temperature_at_z(1)
    #case1.get_nuclear_parameters()
    #if solveConduction:
    #    case1.plotColorMap()

    #case1.plotThermohydraulicParameters([True, True, True, True, True, True])
    
    plotter = plotting([case1])#, case2, case3, case4])
    #plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    plotter.GenFoamComp("BWR\driftFluxModel\THM_prototypeV1\OECD_EX1_12223.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True])

if compute_case_12223:
    #User choice:
    solveConduction = False
    plot_at_z1 = [0.8]

    #Geometric flow parameters:
    flowArea = 107.098 * 10 ** (-6)
    wettedPerimeter = 29.845 * 10 **(-3)
    heatedPerimeter = 54.645 * 10 **(-3)
    clad_radius = heatedPerimeter / 2 * np.pi
    water_radius = ( wettedPerimeter /  2 * np.pi ) - clad_radius
    canal_type1 = "cylindrical" #"square" #"cylindrical"
    canal_width1 = 4.75*10**(-3) #* np.sqrt(Poro)

    print(f'flowArea: {flowArea}, wattedPerimeter: {wettedPerimeter}, heatedPerimeter: {heatedPerimeter}, clad_radius: {clad_radius}, water_radius: {water_radius}')
    print(f'')
    #Geometric fuel parameters:
    fuel_radius1 = 5.6*10**(-3) #External radius of the fuel m
    gap_width1 = 0.54e-3 # m
    clad_width1 = 0.38e-3 # m

    print(f'fuel_radius1: {fuel_radius1}, gap_width1: {gap_width1}, clad_width1: {clad_width1}, clad_radius: {clad_radius}')

    Lf1 = 1.655 # m
    rw1=canal_width1 # canal radius
    gap_rad1 = fuel_radius1+gap_width1
    clad_rad1 = gap_rad1+clad_width1

    #Material properties:
    k_fuel1 = 5 # W/m/K
    H_gap1 = 10000 # W/m^2/K
    k_clad1 = 10 # W/m/K
    
    #Nuclear parameters:
    Qfiss1 = 500000000 # W/m^3
    
    #Meshing parameters:
    I_f1 = 8
    I_c1 = 3
    I_z1 = 10

    #Thermohydraulic parameters:
    T_in1 = 592.75 # K
    u_inlet = 4.467092221 #m/s
    Q_flow1 = u_inlet * 1000 # kg/m^2/s
    P_inlet = 14744900.0 #Pa
    hInlet = IAPWS97(T = T_in1, P = P_inlet * 10**(-6)).h * 1000 #J/kg
    pOutlet = 14719781.65

    #Correlation used:
    voidFractionCorrel = "HEM1"
    
    case1 = Version5_THM_prototype("Testing THM Prototype", water_radius, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "constant", 
                                fuel_radius1, gap_rad1, clad_radius, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, solveConduction, dt=0, t_tot=0, startHeating = 0.1, stopHeating= Lf1, frfaccorel = 'base', voidFractionCorrel = voidFractionCorrel)

    """ print(f"case 1 h_z is {case1.convection_sol.h_z} J/kg")
    print(f"case 1 T_water is {case1.convection_sol.T_water} K")
    print(f"case 1 Hc is {0.5*(case1.convection_sol.Hc[3]+case1.convection_sol.Hc[4])} W/m^2/K")
    print(f"q_fluid1 = {case1.convection_sol.q__}")
    
    print(f"case 1 A_canal = {case1.convection_sol.flowArea} m^2")
    print(f"case 1 T_surf is {case1.convection_sol.T_surf} K") """

    #if solveConduction:
    #    print(f"case 1 T_eff in fuel is {case1.T_eff_fuel} K")
    #    print(f"case 1 T_surf fuel is {case1.T_fuel_surface} K")
    #case1.compare_with_THM_DONJON("C:/Users/cleme/OneDrive/Documents/Poly/BWR/driftFluxModel/THM_prototypeDFM/pincell_mphy_thm_devoir.result",[True, True, True, True, True, True])
    #case1.plot_Temperature_at_z(1)
    #case1.get_nuclear_parameters()
    #if solveConduction:
    #    case1.plotColorMap()

    #case1.plotThermohydraulicParameters([True, True, True, True, True, True])
    
    plotter = plotting([case1])#, case2, case3, case4])
    #plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    plotter.GenFoamComp("BWR\driftFluxModel\THM_prototypeV1\OECD_EX1_12223.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True])

