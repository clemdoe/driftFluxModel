
from THM_main import Version5_THM_prototype
from iapws import IAPWS97
import numpy as np
from THM_main import plotting
import pandas as pd
import matplotlib.pyplot as plt

compute_case_multiphys = True

if compute_case_multiphys:

    case_name = "multiphys"
    #User choice:
    solveConduction = True
    zPlotting = []

    If = 8
    I1 = 3
    # Sensitivity to the meshing parameters
    Iz1 = 75 # number of control volumes in the axial direction, added 70 for comparison with GeN-Foam
    # Iz1 = 10, 20, 40, 50, 70, 80 and 160 are supported for the DONJON solution


    ########## Choice of Thermalhydraulics correlation ##########
    voidFractionCorrel = 'EPRIvoidModel' # 'modBestion', 'HEM1', 'GEramp', 'EPRIvoidModel'
    frfaccorel = "blasius" # 'base', 'blasius', 'Churchill', 'Churchill_notOK' ?
    P2Pcorel = "lockhartMartinelli" # 'base', 'HEM1', 'HEM2', 'MNmodel'
    numericalMethod = "FVM" # "FVM": Solves the system using matrix inversion with preconditioning.
                            # "GaussSiedel" : Applies the Gauss-Seidel iterative solver.
                            # "BiCG" : Uses the BiConjugate Gradient method for solving non-symmetric or indefinite matrices.
                            # "BiCGStab" : Applies the BiCGStab (BiConjugate Gradient Stabilized) method to ensure faster and more stable convergence.

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square" # "square", "cylindrical"
    pitch = 1.295e-2 # m : ATRIUM10 pincell pitch   0.0126 #
    fuelRadius = 0.4435e-2 # m : fuel rod radius
    gapRadius = 0.4520e-2 # m : expansion gap radius : "void" between fuel and clad - equivalent to inner clad radius
    cladRadius = 0.5140e-2 # m : clad external radius
    height = 1.555 # m : height : 3.8 m : active core height in BWRX-300 SMR, 1.555 m : for GeNFoam comparison.


    ## Fluid parameters

    # T_inlet, T_outlet = 270, 287 Celcius
    #tInlet = 270 + 273.15 # K, for BWRX-300 SMR core, try lowering the inlet temperature to set boiling point back and reduce the void fraction increase in the first few cm
    tInlet = 270 + 273.15 # K, for BWRX-300 SMR core
    #Nominal operating pressure = 7.2 MPa (abs)
    pOutlet =  7.2e6 # Pa 
    # Nominal coolant flow rate = 1530 kg/s
    massFlowRate = 1530  / (200*91)  # kg/s

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000 
    #Hgap = 9000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity

    ############ Nuclear Parameters ###########
    # Number of fuel rods and assemblies for a small modular Boiling Water Reactor core
    qFiss_init_1 = [0,0,0,0,0,2.583436219965613782e+08,
4.641112940730458498e+08,
6.431210489477704763e+08,
8.016168845533045530e+08,
9.391810596311589479e+08,
1.051127090849401593e+09,
1.126310112752173901e+09,
1.122408289344516516e+09,
1.081172180443733692e+09,
1.031843496287224889e+09,
9.838948922989957333e+08,
9.376810150227086544e+08,
8.953274315813871622e+08,
8.562025723278212547e+08,
8.190067483864718676e+08,
7.834581455906223059e+08,
7.501008280210506916e+08,
7.185548045340584517e+08,
6.885529786166299582e+08,
6.599259416264193058e+08,
6.325559523337886333e+08,
6.063615523049374819e+08,
5.813109496292912960e+08,
5.573138819593534470e+08,
5.342438041517887115e+08,
5.116967588533892632e+08,
4.901634086280136704e+08,
4.695347716954995394e+08,
4.497400165683849454e+08,
4.307178715644771457e+08,
4.124221828331777453e+08,
3.948140886544378400e+08,
3.778597072997860312e+08,
3.615282028391785026e+08,
3.457908736246812940e+08,
3.306209299694090486e+08,
3.159929161128044724e+08,
3.018824656674641371e+08,
2.882665017080876231e+08,
2.751225698083316684e+08,
2.624294383076381981e+08,
2.501662757233460546e+08,
2.383133177138493061e+08,
2.268512890438538790e+08,
2.157616036733368039e+08,
2.050265648464832902e+08,
1.946292316990556717e+08,
1.845551978268259466e+08,
1.747862106713460684e+08,
1.653070523565417826e+08,
1.561238700595270693e+08,
1.471652876062530279e+08,
1.384405424364709854e+08,
1.299368733217376024e+08,
1.216410632754592896e+08,
1.135397952665630579e+08,
1.056198189602932036e+08,
9.786839536031584442e+07,
9.027184060582201183e+07,
8.281782699442657828e+07,
7.549359329768782854e+07,
6.828624489453729987e+07,
6.118246475390255451e+07,
5.416792984194743633e+07,
4.722621619088541716e+07,
4.033700921450337768e+07,
3.347260770960400254e+07,
2.659163772823368385e+07,
1.962761971753485873e+07,
1.246770629004373029e+07,
]
    
    qFiss_init_8 = [0,0,0,0,0,4.788891356452584267e+06,
8.732191535550605506e+06,
1.235867031913201138e+07,
1.584746567462276295e+07,
1.926153197668480873e+07,
2.262034177054322511e+07,
2.592712287998550013e+07,
2.917906850171817094e+07,
3.237130146764218807e+07,
3.549820539215745032e+07,
3.855399159084783494e+07,
4.153293807561308891e+07,
4.442951738927423954e+07,
4.723835492037649453e+07,
4.995430115753066540e+07,
5.257227606467688084e+07,
5.508765536391407251e+07,
5.749589814773578942e+07,
5.979276364205678552e+07,
6.197420560371338576e+07,
6.403637232046356052e+07,
6.597568998138120025e+07,
6.778879042251381278e+07,
6.947267786767292023e+07,
7.102445102711787820e+07,
7.244152541860800982e+07,
7.372157222911404073e+07,
7.486244050244762003e+07,
7.586237946031786501e+07,
7.671968278864341974e+07,
7.743303323518443108e+07,
7.800130252059832215e+07,
7.842364582488350570e+07,
7.869956292567086220e+07,
7.882855915861797333e+07,
7.881062896569733322e+07,
7.864592241362084448e+07,
7.833363358857159317e+07,
7.787754194844852388e+07,
7.727531312219683826e+07,
7.652929815495108068e+07,
7.564082541500259936e+07,
7.461137889538021386e+07,
7.344283720898020267e+07,
7.213719568725390732e+07,
7.069679981730988622e+07,
6.912417294309805334e+07,
6.742209407778131962e+07,
6.559349230123192072e+07,
6.364156903661230206e+07,
6.156974247011153400e+07,
5.938174203739282489e+07,
5.708139166056744754e+07,
5.467275425687631220e+07,
5.216010950659038126e+07,
4.954801499129451811e+07,
4.684108943086371571e+07,
4.404442397741047293e+07,
4.116278185949420929e+07,
3.820155158363613486e+07,
3.516618279464548081e+07,
3.206218905463650078e+07,
2.889518952822208032e+07,
2.567064497626524791e+07,
2.239323803594372049e+07,
1.906555041048485786e+07,
1.568414862912719138e+07,
1.222957543121803552e+07,
8.639453782079903409e+06,
4.737157247431593947e+06]


    case1 = Version5_THM_prototype("Initialization of BWR Pincell equivalent canal", canalType, pitch, fuelRadius, gapRadius, cladRadius, 
                            height, tInlet, pOutlet, massFlowRate, qFiss_init_1, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting, 
                            solveConduction, dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = voidFractionCorrel, 
                            numericalMethod = numericalMethod)
    
    print(f'U: {case1.convection_sol.U[-1]}')
    print(f'Tsurf : {case1.Tsurf}')
    print(f'Tsurf moy : {np.mean(case1.Tsurf)}')
    plotter = plotting([case1]) #
    genFoamVolumeFraction = 0.494922
    #plotter.plotSimple()
    print(f'Twater : {case1.convection_sol.T_water}')   
    plotter.GenFoamComp(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsPoro\resultMultiPhys.xlsx", 'voidFractionCorrel', [True, True, True, True, True, True], genFoamVolumeFraction)
    plotter.writeResults(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsTransitoire\resultsDFM.xlsx")
    #plotter.compute_error(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsPoro\resultMultiPhys.xlsx", "P2Pcorrel", genFoamVolumeFraction)

def writeGF(GenFoamPathCase, genFoamVolumeFraction):
    # Read the Excel file
    df = pd.read_excel(GenFoamPathCase)

    # Create empty lists for each column
    columns = df.columns.tolist()
    data = [[] for _ in columns]

    # Iterate over each row and append values to the corresponding list
    for index, row in df.iterrows():
        for i, col in enumerate(columns):
            data[i].append(row[col])

    for i in range(len(data[7])):
        data[7][i] = (1/(1-genFoamVolumeFraction)) * data[7][i]

    genfoamCASE = [data[0], data[3], data[7], data[1], data[5]]
    #z water voidfraciton pression vitesse
    return data[0], data[3], data[7], data[1], data[5]

def errors(list1, list2):
    absoluteError = []
    relativeError = []
    RMS = 0
    for i in range(len(list1)):
        absoluteError.append(abs(list1[i] - list2[i]))
        relativeError.append(100*abs(list1[i] - list2[i]) / list2[i])
    
    for i in range(len(absoluteError)):
        RMS += absoluteError[i]**2
    RMS = np.sqrt(RMS/len(absoluteError))

    return absoluteError, relativeError, RMS, np.mean(absoluteError), np.mean(relativeError), max(absoluteError), max(relativeError)


z_GF, Twater_GF, voidFraction_GF, pressure_GF, U_GF= writeGF(r"C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\thermalHydraulicsPoro\resultMultiPhys.xlsx", 0.494922)
Twater_DFM, voidFraction_DFM, pressure_DFM, U_DFM = case1.convection_sol.T_water, case1.convection_sol.voidFraction[-1], case1.convection_sol.P[-1], case1.convection_sol.U[-1]
print(pressure_GF)
absErrorTwater, relErrorTwater, RMSTwater, meanAbsErrorTwater, meanRelErrorTwater, maxAbsErrorTwater, maxRelErrorTwater = errors(Twater_GF, Twater_DFM)
absErrorVoidFraction, relErrorVoidFraction, RMSVoidFraction, meanAbsErrorVoidFraction, meanRelErrorVoidFraction, maxAbsErrorVoidFraction, maxRelErrorVoidFraction = errors(voidFraction_GF, voidFraction_DFM)
absErrorPressure, relErrorPressure, RMSPressure, meanAbsErrorPressure, meanRelErrorPressure, maxAbsErrorPressure, maxRelErrorPressure = errors(pressure_GF, pressure_DFM)


fig1, ax1 = plt.subplots()
ax1.step(z_GF, relErrorTwater, label = 'Twater')
ax1.step(z_GF, relErrorPressure, label = 'Pressure')
ax1.set_xlabel('z [m]')
ax1.set_ylabel('Relative error [%]')
ax1.grid()
ax1.legend()

plotABS = [100*x for x in absErrorVoidFraction]

fig2, ax2 = plt.subplots()
ax2.step(z_GF, plotABS, label = 'Void fraction')
ax2.set_xlabel('z [m]')
ax2.set_ylabel('Absolute error')
ax2.grid()
ax2.legend()


print(f'RMS on Twater : {RMSTwater}')
print(f'Mean relative error on Twater : {meanRelErrorTwater} %')
print(f'Max relative error on Twater : {maxRelErrorTwater} %')
print('\n')
print(f'RMS on voidFraction : {RMSVoidFraction}')
print(f'Mean absolute error on voidFraction : {100*meanAbsErrorVoidFraction} %')
print(f'Max absolute error on voidFraction : {100*maxAbsErrorVoidFraction} %')
print('\n')
print(f'RMS on pressure : {RMSPressure}')
print(f'Mean relative error on pressure : {meanRelErrorPressure} %')
print(f'Max relative error on pressure : {maxRelErrorPressure} %')

plt.show()