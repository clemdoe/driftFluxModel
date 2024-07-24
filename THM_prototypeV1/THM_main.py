#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet, Raphael Guasch


from THM import Version5_THM_prototype

# Begining of the script used to test the THM prototype class.
compute_case1 = True
compute_case2 = False


if compute_case1:
    #Case 1 : base parameters
    # Parameters used to create object from FDM_HeatConductioninFuelpin class
    Qfiss1 = 0.3e9 # W/m^3
    fuel_radius1 = 5.6e-3 # m
    gap_width1 = 0.54e-3 # m
    clad_width1 = 0.38e-3 # m
    k_fuel1 = 5 # W/m/K
    H_gap1 = 10000 # W/m^2/K
    k_clad1 = 10 # W/m/K
    I_f1 = 8
    I_c1 = 3

    # Paramters used to create object from FVM_ConvectionInCanal class
    canal_type1 = "cylindrical" #"square" #"cylindrical"
    canal_width1 = 0.5e-3 # m
    Lf1 = 2 # m
    T_in1 = 500 # K
    Q_flow1 = 7000 # kg/m^2/s
    P_cool1 = 10.8 #MPa
    I_z1 = 10


    rw1=fuel_radius1+gap_width1+clad_width1+canal_width1 # canal radius
    gap_rad1 = fuel_radius1+gap_width1
    clad_rad1 = gap_rad1+clad_width1
    plot_at_z1 = [0.8]
    case1 = Version5_THM_prototype("Case1_ENE6107A_project", rw1, canal_type1, Lf1, T_in1, P_cool1, Q_flow1, I_z1, Qfiss1, "constant", 
                                fuel_radius1, gap_rad1, clad_rad1, k_fuel1, H_gap1, k_clad1, I_f1, I_c1, plot_at_z1, dt=0, t_tot=0)


    print(f"case 1 h_z is {case1.convection_sol.h_z} J/kg")
    print(f"case 1 T_water is {case1.convection_sol.T_water} K")
    print(f"case 1 Hc is {0.5*(case1.convection_sol.Hc[3]+case1.convection_sol.Hc[4])} W/m^2/K")
    print(f"q_fluid1 = {case1.convection_sol.q__}")

    print(f"case 1 A_canal = {case1.convection_sol.flowArea} m^2")
    print(f"case 1 T_surf is {case1.convection_sol.T_surf} K")


    print(f"case 1 T_eff in fuel is {case1.T_eff_fuel} K")
    print(f"case 1 T_surf fuel is {case1.T_fuel_surface} K")
    case1.compare_with_THM_DONJON("C:/Users/cleme/OneDrive/Documents/Poly/BWR/driftFluxModel/THM_prototypeDFM/pincell_mphy_thm_devoir.result",[True, True, True, True, True, True])


