import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from Th_properties import *
from Th_resolution import *
from Linalg_resolution import *
from Th_plotting import plotter

print("Begin of the program")

#Resolution parameters
epsOuterIteration = 1e-3
epsInnerIteration = 1e-3
maxOuterIteration = 1000
maxInnerIteration = 1000
nCells = 20
sousRelaxFactor = 0.12

#Universal constant
g = 9.81 #m/s^2
R = 8.314 #J/(mol*K)

#Geometry parameters
height = 1.655 #m
l = 14.04*10**(-3) #Side of the square fuel rod m
L = 14.04*10**(-3) #Side of the square fuel rod m
fuelRadius = 5.6*10**(-3) #External radius of the fuel m
cladRadius = 6.52*10**(-3) #External radius of the clad m
waterGap = 0.5*10**(-3) #Gap between the clad and the water m
waterRadius =  cladRadius + waterGap #External radius of the water m

cote = 0.0157
Poro = 0.5655077285
flowArea = cote ** 2 * Poro #m2
DV = (height/nCells) * flowArea #Volume of the control volume m3
Area = flowArea #Area of the control volume m2
D_h = Area / (np.pi*cladRadius) #Hydraulic diameter m2
D_h = 0.0078395462 #Hydraulic diameter m2
Dz = height/nCells #Height of the control volume m
zList = np.linspace(0, height, nCells)

#Thermal parameters
K_loss = 0

#Water properties
inletFlowRate = 0

#Boundary conditions
T_inlet = 602.75 #K
P_inlet = 14.9 * (10**6) #Pa
h_inlet = IAPWS97(T = T_inlet, P = P_inlet * 10**(-6)).h * 1000 #J/kg
u_inlet = 4.68292412 #m/s
P_outlet =  14739394.95 #Pa

#Heating parameters
Q = 500000000 #volumetric heat generation rate W/m3
q__fluid = np.pi * fuelRadius**2 * Q #linear heat generation rate W/m
q__ = q__fluid / flowArea #W/m3

#Initial fields
epsilonTarget = 0.18
U, P, H, epsilon_old, rho_old, rho_g_old, rho_l_old, areaMatrix, areaMatrix_1, areaMatrix_2, Dhfg, x_th_old, f, Vgj, Vgj_prime, C0 = setInitialFields(nCells, u_inlet, P_outlet, h_inlet, epsilonTarget, flowArea, D_h, K_loss, DV, Dz, epsInnerIteration, maxInnerIteration)

print(f'U_initial = {U}')
print(f'P_initial = {P}')
print(f'H_initial = {H}')
print(f'epsilon_initial = {epsilon_old}')
print(f'rho_initial = {rho_old}')
print(f'rho_g_initial = {rho_g_old}')
print(f'rho_l_initial = {rho_l_old}')
print(f'x_th_initial = {x_th_old}')

epsilonResidual = []
rhoResidual = []
xThResidual = []
I = []

for k in range(maxOuterIteration):
    print(f'Iteration {k}')
    U,P,H = resolveMVF(U, P, H, epsilon_old, rho_l_old, rho_g_old, rho_old, areaMatrix, areaMatrix_1, areaMatrix_2, Dhfg, Vgj, Vgj_prime, C0, x_th_old,  nCells, u_inlet, P_outlet, h_inlet, DV,g, height, q__)
    epsilon_new, rho_new, rho_g_new, rho_l_new, V_gj_new, Vgj_prime_new, areaMatrix, areaMatrix_1, areaMatrix_2, Dhfg, C0_new, x_th_new, f = updateFields(U, P, H, epsilon_old, rho_old, rho_g_old, rho_l_old, areaMatrix, areaMatrix_1, areaMatrix_2, Dhfg, x_th_old, f, nCells, u_inlet, P_outlet, h_inlet, flowArea, D_h, K_loss, DV, Dz, Vgj, Vgj_prime, C0, epsInnerIteration, maxInnerIteration)
    
    print(f'U = {U}')
    print(f'P = {P}')
    print(f'H = {H}')

    rho_g_relaxed = sousRelaxation(rho_g_new, rho_g_old, sousRelaxFactor)
    rho_l_relaxed = sousRelaxation(rho_l_new, rho_l_old, sousRelaxFactor)
    rho_relaxed = sousRelaxation(rho_new, rho_old, sousRelaxFactor)
    epsilon_relaxed = sousRelaxation(epsilon_new, epsilon_old, sousRelaxFactor)
    x_th_relaxed = sousRelaxation(x_th_new, x_th_old, sousRelaxFactor)
    Vgj_relaxed = sousRelaxation(V_gj_new, Vgj, sousRelaxFactor)
    C0_relaxed = sousRelaxation(C0_new, C0, sousRelaxFactor)
    Vgj_prime_relaxed = sousRelaxation(Vgj_prime_new, Vgj_prime, sousRelaxFactor)
    
    rho_g_residual = np.linalg.norm(np.array(rho_g_relaxed) - np.array(rho_g_old))
    rho_l_residual = np.linalg.norm(np.array(rho_l_relaxed) - np.array(rho_l_old))
    epsilon_residual = np.linalg.norm(np.array(epsilon_relaxed) - np.array(epsilon_old))
    x_th_residual = np.linalg.norm(np.array(x_th_relaxed) - np.array(x_th_old))

    epsilonResidual.append(epsilon_residual)
    rhoResidual.append(rho_g_residual)
    xThResidual.append(x_th_residual)
    I.append(k)

    print(f'rho_g_residual = {rho_g_residual}')
    print(f'rho_l_residual = {rho_l_residual}')
    print(f'epsilon_residual = {epsilon_residual}')
    print(f'x_th_residual = {x_th_residual}')
    
    if epsilon_residual < epsOuterIteration and rho_g_residual < epsOuterIteration and rho_l_residual < epsOuterIteration and x_th_residual < epsOuterIteration:
        print(f'Convergence reached at iteration {k}')
        break
    elif k == maxOuterIteration - 1:
        print('Convergence not reached')
        break
    else:
        epsilon_old = epsilon_relaxed
        rho_old = rho_relaxed
        rho_g_old = rho_g_relaxed
        rho_l_old = rho_l_relaxed
        x_th_old = x_th_relaxed

T = []
for i in range(nCells):
    T.append(IAPWS97(P = P[i] * 10**(-6), h = H[i]/1000, x = 0).T)

plot = plotter(nCells, zList, U, P, T, rho_new, rho_g_new, rho_l_new, x_th_new, H, epsilon_new, epsilonResidual, rhoResidual, xThResidual, I)
plot.plotFields(print_U = True, print_P = True, print_T = True, print_rho = True, print_rho_g = True, print_rho_l = True, print_xth = True, print_H = True, print_epsilon = True, print_epsilon_Res = True, print_rho_Res = True, print_xth_Res = True)
plot.printFields(print_U = True, print_P = True, print_T = True, print_rho = True, print_rho_g = True, print_rho_l = True, print_xth = True, print_H = True, print_epsilon = True, print_epsilon_Res = True, print_rho_Res = True, print_xth_Res = True)
print(zList)
print("End of the program")