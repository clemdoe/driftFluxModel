#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet, Raphael Guasch


from THM import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
# Begining of the script used to test the THM prototype class.
compute_case1 = True
compute_case_openfoam = False

#self.T_in = T_inlet # inlet water temperature K
        #self.P_cool = P_inlet # coolant pressure in MPa, assumed to be constant along the axial profile.
        #IAPWS97(T = self.T_in, P = self.P_cool).h * 1000 #J/kg
        #self.pOutlet =  14739394.95 #Pa


if compute_case1:
    #Case 1 : base parameters
    # Parameters used to create object from FDM_HeatConductioninFuelpin class
    Qfiss1 = 1e9 # W/m^3
    fuel_radius1 = 5.6*10**(-3) #External radius of the fuel m
    gap_width1 = 0.54e-3 # m
    clad_width1 = 0.38e-3 # m
    clad_radius = fuel_radius1 + gap_width1 + clad_width1 #External radius of the clad m
    k_fuel1 = 5 # W/m/K
    H_gap1 = 10000 # W/m^2/K
    k_clad1 = 10 # W/m/K
    I_f1 = 8
    I_c1 = 3

    # Paramters used to create object from FVM_ConvectionInCanal class
    canal_type1 = "square" #"square" #"cylindrical"
    Poro = 0.5655077285
    canal_width1 = 0.0157 * np.sqrt(Poro)
    Lf1 = 1.655 # m
    T_in1 = 602.75 # K
    u_inlet = 4.68292412 #m/s
    Q_flow1 = u_inlet * 1000 # kg/m^2/s
    P_inlet = 14.9 * (10**6) #Pa
    hInlet = IAPWS97(T = T_in1, P = P_inlet * 10**(-6)).h * 1000 #J/kg
    pOutlet = 14739394.95 
    I_z1 = 10


    rw1=fuel_radius1+gap_width1+clad_width1+canal_width1 # canal radius
    gap_rad1 = fuel_radius1+gap_width1
    clad_rad1 = gap_rad1+clad_width1
    plot_at_z1 = [0.8]
    case1 = Version5_THM_prototype("Testing THM Prototype", rw1, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "sinusoidal", 
                                fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, dt=0, t_tot=0)

    print(f"case 1 h_z is {case1.convection_sol.h_z} J/kg")
    print(f"case 1 T_water is {case1.convection_sol.T_water} K")
    print(f"case 1 Hc is {0.5*(case1.convection_sol.Hc[3]+case1.convection_sol.Hc[4])} W/m^2/K")
    print(f"q_fluid1 = {case1.convection_sol.q__}")

    print(f"case 1 A_canal = {case1.convection_sol.flowArea} m^2")
    print(f"case 1 T_surf is {case1.convection_sol.T_surf} K")


    print(f"case 1 T_eff in fuel is {case1.T_eff_fuel} K")
    print(f"case 1 T_surf fuel is {case1.T_fuel_surface} K")
    #case1.compare_with_THM_DONJON("C:/Users/cleme/OneDrive/Documents/Poly/BWR/driftFluxModel/THM_prototypeDFM/pincell_mphy_thm_devoir.result",[True, True, True, True, True, True])
    #case1.plot_Temperature_at_z(1)
    case1.get_nuclear_parameters()
    case1.plotColorMap()
    case1.plotThermohydraulicParameters([True, True, True, True, True])


if compute_case_openfoam:
    #Case 1 : base parameters
    # Parameters used to create object from FDM_HeatConductioninFuelpin class
    Qfiss1 = 1943301220 # W/m^3
    

    H_gap1 = 10000 # W/m^2/K
    k_clad1 = 10 # W/m/K
    I_f1 = 8
    I_c1 = 3
    k_fuel1 = 5 # W/m/K


    # Paramters used to create object from FVM_ConvectionInCanal class
    canal_type1 = "square" #"square" #"cylindrical"

    fuel_radius1 = 5.6*10**(-3) #External radius of the fuel m
    gap_width1 = 0.54e-3 # m
    clad_width1 = 0.38e-3 # m
    clad_radius = fuel_radius1 + gap_width1 + clad_width1 #External radius of the clad m
    
    Poro = 0.5655077285   #ok 
    canal_width1 = 0.0126 * np.sqrt(Poro)  #ok 
    Lf1 = 2.155 # m   ok 

    T_in1 = 592.75 # K    ok
    u_inlet = 4.467092221 #m/s    ok
    Q_flow1 = u_inlet * 1000 # kg/m^2/s
    P_inlet = 14771400.0 #Pa ok
    hInlet = IAPWS97(T = T_in1, P = P_inlet * 10**(-6)).h * 1000 #J/kg ok 
    pOutlet = 14719781.65  #ok 
    I_z1 = 10  #ok 


    rw1=fuel_radius1+gap_width1+clad_width1+canal_width1 # canal radius
    gap_rad1 = fuel_radius1+gap_width1
    clad_rad1 = gap_rad1+clad_width1
    plot_at_z1 = [0.8]
    case1 = Version5_THM_prototype("Testing THM Prototype VS GenFoam exercice 1", rw1, canal_type1, Lf1, hInlet, pOutlet, Q_flow1, I_z1, Qfiss1, "linear", 
                                fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, dt=0, t_tot=0)

    print(f"case 1 h_z is {case1.convection_sol.h_z} J/kg")
    print(f"case 1 T_water is {case1.convection_sol.T_water} K")
    print(f"case 1 Hc is {0.5*(case1.convection_sol.Hc[3]+case1.convection_sol.Hc[4])} W/m^2/K")
    print(f"q_fluid1 = {case1.convection_sol.q__}")

    print(f"case 1 A_canal = {case1.convection_sol.flowArea} m^2")
    print(f"case 1 T_surf is {case1.convection_sol.T_surf} K")


    print(f"case 1 T_eff in fuel is {case1.T_eff_fuel} K")
    print(f"case 1 T_surf fuel is {case1.T_fuel_surface} K")
    #case1.compare_with_THM_DONJON("C:/Users/cleme/OneDrive/Documents/Poly/BWR/driftFluxModel/THM_prototypeDFM/pincell_mphy_thm_devoir.result",[True, True, True, True, True, True])
    #case1.plot_Temperature_at_z(1)
    case1.get_nuclear_parameters()
    case1.plotColorMap()
    case1.plotThermohydraulicParameters([True, True, True, True, True])
