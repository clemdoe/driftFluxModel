
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
    voidFractionCorrel = 'EPRIvoidCorrel' # 'modBestion', 'HEM1', 'GEramp', 'EPRIvoidModel'
    frfaccorel = "Churchill" # 'base', 'blasius', 'Churchill', 'Churchill_notOK' ?
    P2Pcorel = "lockhartMartinelli" # 'base', 'HEM1', 'HEM2', 'MNmodel', "lockhartMartinelli"
    numericalMethod = "BiCG" # "FVM": Solves the system using matrix inversion with preconditioning.
                            # "GaussSiedel" : Applies the Gauss-Seidel iterative solver.
                            # "BiCG" : Uses the BiConjugate Gradient method for solving non-symmetric or indefinite matrices.
                            # "BiCGStab" : Applies the BiCGStab (BiConjugate Gradient Stabilized) method to ensure faster and more stable convergence.

    ########## Thermal hydraulics parameters ##########
    ## Geometric parameters
    canalType = "square" # "square", "cylindrical"
    pitch =1.295e-2 #1.295e-2 # m : ATRIUM10 pincell pitch   0.0126 #
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
    massFlowRate = 8.407 * 10**(-2) #1530  / (200*91)  # kg/s

    ## Material parameters
    kFuel = 4.18 # W/m.K, TECHNICAL REPORTS SERIES No. 59 : Thermal Conductivity of Uranium Dioxide, IAEA, VIENNA, 1966
    Hgap = 10000 
    #Hgap = 9000
    kClad = 21.5 # W/m.K, Thermal Conductivity of Zircaloy-2 (as used in BWRX-300) according to https://www.matweb.com/search/datasheet.aspx?MatGUID=eb1dad5ce1ad4a1f9e92f86d5b44740d
    # k_Zircaloy-4 = 21.6 W/m.K too so check for ATRIUM-10 clad material but should have the same thermal conductivity

    ############ Nuclear Parameters ###########
    # Number of fuel rods and assemblies for a small modular Boiling Water Reactor core
    qFiss_init_1 = [
    0, 0, 0, 0, 0,
    2.572889753855869770e+08, 4.621927967787369490e+08, 6.406415911726634502e+08,
    7.989263551705286503e+08, 9.368073822130444050e+08, 1.050040830184697986e+09,
    1.128633189898637533e+09, 1.136367294688590288e+09, 1.095419047087164402e+09,
    1.044077912759370923e+09, 9.943807092254955769e+08, 9.461677210411007404e+08,
    9.021576678984748125e+08, 8.616554852576858997e+08, 8.235748235256036520e+08,
    7.870193731451634169e+08, 7.528343987596774101e+08, 7.205868189580025673e+08,
    6.899854820891946554e+08, 6.608385695487705469e+08, 6.330128220975966454e+08,
    6.064141979302755594e+08, 5.809981883720041513e+08, 5.566667498384325504e+08,
    5.334158805506106019e+08, 5.106185906750129461e+08, 4.888616743987929225e+08,
    4.680418413607435822e+08, 4.480823018691839576e+08, 4.289178713913894892e+08,
    4.104999950093899369e+08, 3.927875433283962607e+08, 3.757450783725664616e+08,
    3.593401190360649228e+08, 3.435428298335876465e+08, 3.283252205445634127e+08,
    3.136608127315912247e+08, 2.995245730441089272e+08, 2.858924241120778322e+08,
    2.727413334744169116e+08, 2.600492691147776246e+08, 2.477949993726033866e+08,
    2.359580040147083104e+08, 2.245187410205347240e+08, 2.134581352437432706e+08,
    2.027580452864131629e+08, 1.924011078743061721e+08, 1.823721384795089960e+08,
    1.726517507062035501e+08, 1.632237818138405383e+08, 1.540736030771359503e+08,
    1.452067280582028925e+08, 1.365544262035239339e+08, 1.281253791502125561e+08,
    1.199069257143074572e+08, 1.118859711857886612e+08, 1.040493875064795911e+08,
    9.638441344794103503e+07, 8.887738738146932423e+07, 8.151579818981066346e+07,
    7.428683462228643894e+07, 6.717751868743330240e+07, 6.017448333194191754e+07,
    5.326333326763391495e+07, 4.642765008474908769e+07, 3.964716921931123734e+07,
    3.289436176693907380e+07, 2.612832065389018878e+07, 1.928348757017016783e+07,
    1.224879230723222345e+07
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

    qFiss_init_2 = [0,0,0,0,0,6.756780329588347673e+07,
1.225511900859650373e+08,
1.721201588364999592e+08,
2.184756990877254605e+08,
2.622164325068090856e+08,
3.033417810590132475e+08,
3.416546348987942934e+08,
3.769016814184232950e+08,
4.088274292638525367e+08,
4.371881700968404412e+08,
4.617434414665185809e+08,
4.822272584651400447e+08,
4.982545610289983749e+08,
5.090542952822619677e+08,
5.128151671423504353e+08,
5.028908442671745420e+08,
4.816494460485277772e+08,
4.594450419569328427e+08,
4.375400598594470024e+08,
4.159057314118189216e+08,
3.953037285414503813e+08,
3.757576172799536586e+08,
3.572386292830754519e+08,
3.397147725512966514e+08,
3.229261537737228274e+08,
3.069688156452266574e+08,
2.918536963616145849e+08,
2.775041173917204738e+08,
2.638685891796860397e+08,
2.508966003822812438e+08,
2.385466214268116951e+08,
2.267815691616494060e+08,
2.155683844462197423e+08,
2.048766982246900797e+08,
1.946787425975481272e+08,
1.849503512663356066e+08,
1.756663352557563484e+08,
1.668055073694643378e+08,
1.583327186489859819e+08,
1.501975244474098980e+08,
1.424295156968826056e+08,
1.350023918168404102e+08,
1.278944542725161612e+08,
1.210865389891439080e+08,
1.145611492998585105e+08,
1.083020779998999685e+08,
1.022940071687185913e+08,
9.652241924155846238e+07,
9.097347473287303746e+07,
8.563391219185398519e+07,
8.049108710861487687e+07,
7.553284407958579063e+07,
7.074753348159296811e+07,
6.612389475330245495e+07,
6.165119534588532895e+07,
5.731893058958909661e+07,
5.311704601479417831e+07,
4.903593179398689419e+07,
4.506573354649534076e+07,
4.119738613138597459e+07,
3.742183274476308376e+07,
3.373002214075976610e+07,
3.011286972534953058e+07,
2.656079624016347900e+07,
2.306309831600875035e+07,
1.960652006010331959e+07,
1.617248237995365076e+07,
1.273136933229391277e+07,
9.230969512592667714e+06,
5.572810326342738234e+06]
    print(f'len qFiss_init_2 : {len(qFiss_init_2)}')

    case1 = Version5_THM_prototype("Initialization of BWR Pincell equivalent canal", canalType, pitch, fuelRadius, gapRadius, cladRadius, 
                            height, tInlet, pOutlet, massFlowRate, qFiss_init_2, kFuel, Hgap, kClad, Iz1, If, I1, zPlotting, 
                            solveConduction, dt = 0, t_tot = 0, frfaccorel = frfaccorel, P2Pcorel = P2Pcorel, voidFractionCorrel = 'EPRIvoidModel',
                            numericalMethod = numericalMethod)
    
    print(f'U: {case1.convection_sol.U[-1]}')
    print(f'Tsurf : {case1.Tsurf}')
    print(f'Tsurf moy : {np.mean(case1.Tsurf)}')
    plotter = plotting([case1]) #
    genFoamVolumeFraction = 0.494922
    plotter.plotSimple()
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