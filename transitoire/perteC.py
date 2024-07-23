#Drift flux model for BWR fuel assembly class
# Used to calculate the drift velocity of the mixture in the fuel assembly
# Authors : Clément Huet

from class_MVF import FVM
import numpy as np
from iapws import IAPWS97
import csv
import matplotlib.pyplot as plt

##Creation of function
#FUnction to calculate the temperature of the mixture
def getTemperature(P, H):
    T = IAPWS97(P = P, h = H).T
    return T

#Function to calculate the density of the mixture
def getDensity(epsilon, H, P):
    #print(f'H: {H}, P: {P}')
    T = getTemperature(P, H)
    state = IAPWS97(T = T, x = epsilon)
    rho_g = state.Vapor.rho
    rho_l = state.Liquid.rho
    rho = rho_g * (1 - epsilon) + rho_l * epsilon
    return rho_g, rho_l, rho

#Function to calculate the areas of the mixture
def getAreas(A, Phi2Phi, f, D_h, K_loss, DV, Dz):
    A_chap = A #+  (Phi2Phi/2) * ((f / D_h) + (K_loss / Dz)) * DV
    return A_chap

#Function to calculate the thermodynamic quality of the mixture
def getThermodynamicQuality(U, H, P):
    T = getTemperature(P, H)
    gas = IAPWS97(T = T, x = 0)
    liquid = IAPWS97(T = T, x = 1)
    h_g_sat = gas.h
    h_l_sat = liquid.h
    h_m = IAPWS97(P = P, h = H).h
    x_th = h_m - h_l_sat / h_g_sat - h_l_sat #h_l_sat, h_g_sat are the specific enthalpies of the liquid and gas phases at saturation temperature
    return x_th

#Function to calculate the drift velocity of the mixture
def getDriftVelocity(rho_g, rho_l, g, D_h):
    return 0.188 * np.sqrt(((rho_l - rho_g) * g * D_h ) / rho_g )

#Function to calculate the constant C0 called the distribution parameter
def getC0(rho_g, rho_l):
    return 1.2 - 0.2*np.sqrt(rho_g / rho_l)

#Function to calculate the void fraction of the mixture
def getVoidFraction(rho_g, rho_l, U, H, P, U_old, rho, D_h, g):
    C0 = getC0(rho_g, rho_l)
    x_th = getThermodynamicQuality(U, H, P)
    Vgj = getDriftVelocity(rho_g, rho_l, g, D_h)
    Vgj_prime = Vgj + (C0 -1) * U_old
    epsilon = x_th / (C0 * (x_th + (rho_g/rho_l) * (1 - x_th)) + (rho_g * Vgj_prime / rho * U))
    return epsilon

#Function to calculate the hydraulic diameter
def getD_h(L,l,geoType,cladRadius,Phi):
    if geoType == "square":
        return ((l*L)-(np.pi*cladRadius**2))/(cladRadius*2*np.pi)
    if geoType =="cylinder":
        print('Cylinder geometry not implemented yet')
        return
    
#Function to merge the 3 component of the matrix U,P,H
def createVar(U,P,H):
    var = np.zeros(3*len(U))
    for i in range(len(var)):
        if i < len(U):
            var[i] = U[i]
        elif i < 2*len(U):
            var[i] = P[i-len(U)]
        else:
            var[i] = H[i-2*len(U)]
    return var

#Function to split the 3 component of the matrix U,P,H
def splitVar(var):
    U = var[:len(var)//3]
    P = var[len(var)//3:2*len(var)//3]
    H = var[2*len(var)//3:]
    return U,P,H

## Parameters of the system
# Equation resolution parameters
eps = 10**(-3)
N_iterations = 1000

# Constant of the problem
sizeMesh = 4 #Number of volumes for the discretization using the FVM class
N_vol = 12 #Number of volumes for the discretization using the FVM class
Phi = 1 #Porosity
Height = 2 #Height of the fuel rod m
l = 14.04*10**(-3) #Side of the square fuel rod m
L = 14.04*10**(-3) #Side of the square fuel rod m
Phi2Phi = 0.5 #Two-phase friction multiplier ???????,
f = 0.5 #correction factor for drag coefficient ???????
K_loss = 0 #loss coefficient ???????
g = 9.81 #gravity m/s2
q__ = 300000 #volumetric heat generation rate W/m3
cladRadius = 6.52*10**(-3) #External radius of the clad m
waterGap = 0.5*10**(-3) #Gap between the clad and the water m
waterRadius =  cladRadius + waterGap #External radius of the water m
massFlowRate = 7000 #kg/m2/s

#Initial/boundary conditions of the system
rho_l_start= 1000 #kg/m3
rho_g_start = 1 #kg/m3
epsilon_start = 0
U_start = 7 #m/2
T_inlet = 500 #K
P_outlet = 10.800000 #MPa
P_inlet = 10.800000 #MPa
h_start = IAPWS97(T = T_inlet, P = P_inlet).h #J/kg

#Calulated values
DV = (Height/N_vol)*((l*L)-(np.pi*cladRadius**2)) #Volume of the control volume m3
Area = ((np.pi*waterRadius**2)-(np.pi*cladRadius**2)) #Area of the control volume m2
U_start = massFlowRate / (rho_l_start) #m/s
#Area = 2
D_h = getD_h(L,l,"square",cladRadius,Phi) #Hydraulic diameter m2
Dz = Height/N_vol #Height of the control volume m

#print(f"DV: {DV}, Area: {Area}, D_h: {D_h}, Dz: {Dz}, Height: {Height}, L: {L}, l: {l}, rho_l_start: {rho_l_start}, rho_g_start: {rho_g_start}, epsilon_start: {epsilon_start}, U_start: {U_start}, T_inlet: {T_inlet}, P_outlet: {P_outlet}, P_inlet: {P_inlet}, h_start: {h_start}")

""" #PATHS Values
inletFlowRate = 9.506 #m3/s
Power = 0
outletPressure = 7.18 #MPa
FlowArea = 0.010334 #m2
Height = 4.53*10**(-3) #m

#Calculated PATHS values
DV = (Height/N_vol) * FlowArea #Volume of the control volume m3
Area = FlowArea #Area of the control volume m2
D_h = Area / np.sqrt(Area)
Dz = Height/N_vol #Height of the control volume m
U_start = inletFlowRate * Area / rho_l_start #m/s """

V_gj_start = getDriftVelocity(rho_g_start, rho_l_start, g, D_h) #m/s
C0_start = getC0(rho_g_start, rho_l_start)
#Vgj_prime_start = V_gj_start + (C0_start -1) * U_start #m/s
Vgj_prime_start = V_gj_start + (C0_start -1) * U_start #m/s
rho_start = rho_g_start * epsilon_start + rho_l_start * (1 - epsilon_start)
Dhfg_start = 500 #kJ/kg specific enthalpy of vaporization
h_inlet = IAPWS97(T = T_inlet, P = P_inlet).h #J/kg

#Initial fields of the system
U = np.ones(sizeMesh)
P = np.ones(sizeMesh)
H = np.ones(sizeMesh)
rho_g_old = np.ones(N_vol)*rho_g_start
rho_l_old = np.ones(N_vol)*rho_l_start
rho_old = np.ones(N_vol)*rho_start
V_gj_old = np.ones(N_vol)*V_gj_start
Vgj_prime = np.ones(N_vol)*Vgj_prime_start
epsilon_old = np.ones(N_vol)*epsilon_start
areaMatrix = np.ones(N_vol)*Area
areaMatrix_old_ = [getAreas(areaMatrix[i], Phi2Phi, f, D_h, K_loss, DV, Dz) for i in range(N_vol)]
Dhfg = np.ones(N_vol)*Dhfg_start

C0 = np.ones(N_vol)*C0_start
x_th = np.ones(N_vol)
T = np.ones(N_vol)

print(f'rho_old: {rho_old}, Area : {Area})')

for j in range(N_iterations):

    print(f"\n Begin itération number: {j}")
    U_old = U
    P_old = P
    H_old = H

    print(f'U_old: {U_old}, P_old: {P_old}, H_old: {H_old}')

    VAR_old = createVar(U_old,P_old,H_old)

    DM1 = (1/2) * (P[-1]*areaMatrix[-1] - P[-2]*areaMatrix[-2]) * ( (U_old[-1]) + (U_old[-2]) )
    VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = - rho_old[-2] * VAR_old[sizeMesh-2] * areaMatrix_old_[-2], Am1 = rho_old[-1] * VAR_old[sizeMesh-1] * areaMatrix_old_[-1], D0 = U_start, Dm1 = DM1, N_vol = 12, H = Height)
    VAR_VFM_Class.boundaryFilling()
    for i in range(1, VAR_VFM_Class.N_vol-1):
        #Inside the velocity submatrix
        if i < sizeMesh-1:
            VAR_VFM_Class.set_ADi(i, ci = - rho_old[i-1]*areaMatrix[i-1],
            ai = rho_old[i]*areaMatrix[i],
            bi = 0,
            di =  0)
        elif i == sizeMesh-1:
            VAR_VFM_Class.set_ADi(i, 
            ci = - rho_old[i-1]*areaMatrix[i-1],
            ai = rho_old[i]*areaMatrix[i],
            bi = 0,
            di =  0)

        #Inside the pressure submatrix
        elif i == sizeMesh:
            print(f'areaMatrix[i]: {areaMatrix[i]}, areaMatrix[i+1]: {areaMatrix[i+1]}, rho_old[i]: {rho_old[i]}, rho_old[i+1]: {rho_old[i+1]}')
            VAR_VFM_Class.set_ADi(sizeMesh, 
            ci = 0,
            ai = - areaMatrix[i],
            bi = areaMatrix[i+1],
            di =  ((rho_old[i+1]- rho_old[i])* g * DV / 2))
        
            VAR_VFM_Class.fillingOutsideBoundary(i, i-sizeMesh,
            ai = - rho_old[i]*VAR_old[i-sizeMesh]*areaMatrix_old_[i],
            bi = rho_old[i+1]*VAR_old[i-sizeMesh]*areaMatrix_old_[i+1])

        elif i > sizeMesh and i < 2*sizeMesh-1:
            print(f'areaMatrix[i]: {areaMatrix[i]}, areaMatrix[i+1]: {areaMatrix[i+1]}, rho_old[i]: {rho_old[i]}, rho_old[i+1]: {rho_old[i+1]}, areaMatrix_old_[i]: {areaMatrix_old_[i]}, areaMatrix_old_[i+1]: {areaMatrix_old_[i+1]}')
            VAR_VFM_Class.set_ADi(i, ci = 0,
            ai = - areaMatrix[i],
            bi = areaMatrix[i+1],
            di = ((rho_old[i+1]- rho_old[i])* g * DV / 2))
        
            print(f'rho_old[i]: {rho_old[i]}, VAR_old[i-sizeMesh]: {VAR_old[i-sizeMesh]}, areaMatrix_old_[i]: {areaMatrix_old_[i]}, rho_old[i+1]: {rho_old[i+1]}, VAR_old[i+1-sizeMesh]: {VAR_old[i+1-sizeMesh]}, areaMatrix_old_[i+1]: {areaMatrix_old_[i+1]}')
            VAR_VFM_Class.fillingOutsideBoundary(i, i-sizeMesh,
            ai = - rho_old[i]*VAR_old[i-sizeMesh]*areaMatrix_old_[i],
            bi = rho_old[i+1]*VAR_old[i+1-sizeMesh]*areaMatrix_old_[i+1])

        elif i == 2*sizeMesh -1:
            VAR_VFM_Class.set_ADi(i, 
            ci = 0,
            ai = 1,
            bi = 0,
            di =  P_outlet)

            VAR_VFM_Class.fillingOutsideBoundary(2*sizeMesh -1, 2*sizeMesh -1 - sizeMesh,
            ai = 0,
            bi = 0)

        #Inside the enthalpy submatrix
        elif i == 2*sizeMesh:
            VAR_VFM_Class.set_ADi(2*sizeMesh, 
            ci = 0,
            ai = 1,
            bi = 0,
            di =  h_inlet)

        elif i > 2*sizeMesh and i < 3*sizeMesh:
            DI = (1/2) * (VAR_old[i-sizeMesh]*areaMatrix[i] - VAR_old[i-1-sizeMesh]*areaMatrix[i-1]) * ( (VAR_old[i-2*sizeMesh] ) + (VAR_old[i-1-2*sizeMesh]) )
            VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * VAR_old[i-1-sizeMesh] * areaMatrix_old_[i-1],
            ai = rho_old[i] * VAR_old[sizeMesh] * areaMatrix_old_[i],
            bi = 0,
            di =  DI)

    print(j, VAR_VFM_Class.A, VAR_VFM_Class.D)
    VAR = VAR_VFM_Class.resoudre_h()
    U, P, H = splitVar(VAR)

    for i in range(VAR_VFM_Class.N_vol):
        #rho_g_old[i], rho_l_old[i], rho_old[i] = getDensity(epsilon_old[i], H[i], P[i])
        #print(f'rho_g_old: {rho_g_old[i]}, rho_l_old: {rho_l_old[i]}, rho_old: {rho_old[i]}')
        pass

    U_residual = np.linalg.norm(U - U_old)
    P_residual = np.linalg.norm(P - P_old)
    H_residual = np.linalg.norm(H - H_old)

    if (np.linalg.norm(U - U_old) < eps) and (np.linalg.norm(P - P_old) < eps) and (np.linalg.norm(H - H_old) < eps):
        print(f"Itération number: {j}, U_residual: {U_residual}, P_residual: {P_residual}, H_residual: {H_residual}")
        print(f"Convergence reached at iteration {j}")
        print(f'U: {U}, P: {P}, H: {H}')
        break

    elif j == N_iterations - 1:
        print(f'rho_old: {rho_old}, Area : {Area})')
        raise ValueError("The system did not converge")
        
    else:
        print(f"Itération number: {j}, U_residual: {U_residual}, P_residual: {P_residual}, H_residual: {H_residual}")
        print(f"Convergence not reached yet at iteration {j}")

""" # Assuming you have the arrays U, P, and H
# Define the file path
csv_file = 'output.csv'

# Combine the arrays into a list of tuples
data = list(zip(U, P, H))

# Write the data to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['U', 'P', 'H'])  # Write the header
    writer.writerows(data)  # Write the data rows
 """

print(f'rho_old: {rho_old}, Area : {Area})')