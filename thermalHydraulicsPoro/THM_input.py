#Used to run the THM prototype class and compare the results with a reference THM_DONJON case.
#Authors : Clement Huet

from THM_main import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
from THM_main import plotting
import pandas as pd
import matplotlib.pyplot as plt

# Begining of the script used to test the THM prototype class.


compute_case_genfoam_NEW_Ex1_12223  = False
compute_case_genfoam_NEW_Ex1_12223_transient = False
compute_case_genfoam_comparaison_nCells = False
compute_case_multiphys = True
compute_case_correlDFM = False


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
    P2Pcorel = "HEM2"                       #choice between 'HEM1' and 'HEM2' and 'MNmodel'
    numericalMethod = 'FVM'            #choice between 'BiCG', 'BiCGStab', 'GaussSiedel' and 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss = 1943301220
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


if compute_case_genfoam_comparaison_nCells:
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

        
    Iz1 = 10
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
    
    Iz1 = 20
    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
    case2 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    
    Iz1 = 25
    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
    case3 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    Iz1 = 50
    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
    case4 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    Iz1 = 100
    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
    case4 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    Iz1 = 2000
    Qfiss1 = []
    for i in range(Iz1): 
        if i*(height/Iz1) < 0.1:
            Qfiss1.append(0)
        else:
            Qfiss1.append(qFiss)
    print(Qfiss1)
    case4 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)


    plotter = plotting([case1, case2, case3, case4])
    #plotter.plotComparison("numericalMethod", [True, True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'nCells', [True, True, True, True, True, True], genFoamVolumeFraction)
    plotter.writeResults(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\resultsDFM.xlsx")
    plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", "nCells", genFoamVolumeFraction)

if compute_case_genfoam_NEW_Ex1_12223_transient:

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
    Iz1 = 100 # number of control volumes in the axial direction

    ## Thermalhydraulics correlation
    voidFractionCorrel = "EPRIvoidModel"    #choice between 'EPRIvoidModel' and 'GEramp' and 'modBestion' and 'HEM1'
    frfaccorel = "base"                #choice between 'Churchill' and 'blasius'
    P2Pcorel = "HEM1"                       #choice between 'HEM1' and 'HEM2' and 'MNmodel'
    numericalMethod = 'FVM'            #choice between 'BiCG', 'BiCGStab', 'GaussSiedel' and 'FVM'

    ############ Nuclear Parameters ###########
    ## Fission parameters
    qFiss =  1943301220 #W/m3
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
                 dt = 0.1, t_tot = 4, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)
    frfaccorel = "Churchill"
    case3 = Version5_THM_prototype(case_name, canalType,
                 waterRadius, fuelRadius, gapRadius, cladRadius, height, tInlet, pOutlet, qFlow, Qfiss1,
                 kFuel, Hgap, kClad, Iz1, If, I1, plot_at_z1, solveConduction,
                 dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, numericalMethod = numericalMethod)

    plotter = plotting([case2, case3])#
    #plotter.plotComparison("voidFractionCorrel", [True, True, True, True, True, True])
    genFoamVolumeFraction = 0.5655077285
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\results.xlsx", 'frfaccorel', [True, True, True, True, True, True], genFoamVolumeFraction)

    # Tracé de chaque courbe pour les différents pas de temps
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    for j in range(len(case2.convection_sol.voidFractionList)):
        if j % 5 == 0:
            ax1.plot(case2.convection_sol.z_mesh, case2.convection_sol.voidFractionList[j], label=f'Temps {case2.convection_sol.timeList[j]}')
            ax2.plot(case2.convection_sol.z_mesh, case2.convection_sol.velocityList[j], label=f'Temps {case2.convection_sol.timeList[j]}')
            ax3.plot(case2.convection_sol.z_mesh, case2.convection_sol.pressureList[j], label=f'Temps {case2.convection_sol.timeList[j]}')
            ax4.plot(case2.convection_sol.z_mesh, case2.convection_sol.enthalpyList[j], label=f'Temps {case2.convection_sol.timeList[j]}')

    # Configuration du graphique
    ax1.title.set_text(f'Iz : {Iz1}, dt : {case2.dt}, t_tot : {case2.t_end}, Qfiss : {qFiss}')
    ax1.set_xlabel('Hauteur')
    ax1.set_ylabel('Fraction de vide')
    ax1.legend(loc="upper left")
    ax1.grid()

    ax2.title.set_text(f'Iz : {Iz1}, dt : {case2.dt}, t_tot : {case2.t_end}, Qfiss : {qFiss}')
    ax2.set_xlabel('Hauteur')
    ax2.set_ylabel('Velocity')
    ax2.legend(loc="upper left")
    ax2.grid()

    ax3.title.set_text(f'Iz : {Iz1}, dt : {case2.dt}, t_tot : {case2.t_end}, Qfiss : {qFiss}')
    ax3.set_xlabel('Hauteur')
    ax3.set_ylabel('Pressure')
    ax3.legend(loc="upper left")
    ax3.grid()
    
    ax4.title.set_text(f'Iz : {Iz1}, dt : {case2.dt}, t_tot : {case2.t_end}, Qfiss : {qFiss}')
    ax4.set_xlabel('Hauteur')
    ax4.set_ylabel('Enthalpy')
    ax4.legend(loc="upper left")
    ax4.grid()
    
    plt.show()


    
if compute_case_correlDFM:
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