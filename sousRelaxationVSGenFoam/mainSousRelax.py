#Drift flux model for BWR fuel assembly class
# Used to calculate the drift velocity of the mixture in the fuel assembly
# Authors : Clément Huet

from class_MVF import FVM
import numpy as np
from iapws import IAPWS97
import csv
import matplotlib.pyplot as plt
from function import *
import cProfile

def main():
    print("Begin of the program")

    ## Parameters of the system
    # Equation resolution parameters
    eps = 10**(-3)
    N_iterations = 250
    sousRelaxFactor = 1

    # Constant of the problem
    sizeMesh = 30 #Number of volumes for the discretization using the FVM class 
    ####WARNING CHANGE THIS VALUE IN EPSILON ARRAY
    N_vol = sizeMesh * 3 #Number of volumes for the discretization using the FVM class
    Phi = 1 #Porosity
    Height = 2 #Height of the fuel rod m
    l = 14.04*10**(-3) #Side of the square fuel rod m
    L = 14.04*10**(-3) #Side of the square fuel rod m
    fuelRadius = 5.6*10**(-3) #External radius of the fuel m
    cladRadius = 6.52*10**(-3) #External radius of the clad m
    waterGap = 0.5*10**(-3) #Gap between the clad and the water m
    waterRadius =  cladRadius + waterGap #External radius of the water m
    g = 9.81 #gravity m/s2
    K_loss = 0 #loss coefficient

    """ #Calulated values with dimension (not given in PATHS)
    Area = ((np.pi*waterRadius**2)-(np.pi*cladRadius**2)) #Area of the control volume m2
    print(f'Area 1: {Area} m2')
    DV = (Height/sizeMesh)*Area #Volume of the control volume m3
    U_start = massFlowRate / (rho_l_start) #m/s
    #Area = 2
    D_h = Area / (np.pi*cladRadius) #Hydraulic diameter m2
    #D_h = getD_h(L,l,"square",cladRadius,Phi) #Hydraulic diameter m2
    Dz = Height/sizeMesh #Height of the control volume m """

    #GeNFoam values
    P_outlet = 14739394.95 #Pa
    cote = 0.0157
    Poro = 0.5655077285
    FlowArea = cote ** 2 * Poro #m2
    Height = 1.655 #m
    Q = 500000000 #300000000 #volumetric heat generation rate W/m3
    q__fluid = np.pi * fuelRadius**2 * Q #linear heat generation rate W/m
    q__ = q__fluid / FlowArea #W/m3
    T_inlet = 602.75 #K
    P_inlet = 14.9 * (10**6) #Pa
    h_inlet = IAPWS97(T = T_inlet, P = P_inlet * 10**(-6)).h * 1000 #J/kg

    #boundary conditions
    epsilon_inlet=  0.000001
    rho_l_inlet= 1000 #kg/m3
    rho_g_inlet = 916.8 #kg/m3
    rho_inlet = rho_l_inlet * (1 - epsilon_inlet) + rho_g_inlet * epsilon_inlet
    phi2phi_inlet = getPhi2Phi(epsilon_inlet)
        
    #Calculated values
    DV = (Height/sizeMesh) * FlowArea #Volume of the control volume m3
    Area = FlowArea #Area of the control volume m2
    D_h = Area / (np.pi*cladRadius) #Hydraulic diameter m2
    D_h = 0.0078395462 #Hydraulic diameter m2
    Dz = Height/sizeMesh #Height of the control volume m
    #U_inlet = inletFlowRate * (Area / rho_l_inlet) #m/s
    #U_inlet= inletFlowRate / rho_l_inlet #m/s
    U_inlet = 4.68292412 #m/s
    zList = [i*Dz for i in range(sizeMesh)]

    #Initial fields of the system
    U = np.ones(sizeMesh)*U_inlet
    P = np.ones(sizeMesh)*P_outlet
    H = np.ones(sizeMesh)*h_inlet
    epsilon = np.array([i*0.8/sizeMesh for i in range(sizeMesh)])

    rho = np.ones(sizeMesh)
    rho_g = np.ones(sizeMesh)
    rho_l = np.ones(sizeMesh)
    V_gj = np.ones(sizeMesh)
    Vgj_prime = np.ones(sizeMesh)
    areaMatrix = np.ones(sizeMesh)
    areaMatrix_1 = np.ones(sizeMesh)
    areaMatrix_2 = np.ones(sizeMesh)
    Dhfg = np.ones(sizeMesh)
    C0 = np.ones(sizeMesh)
    x_th = np.ones(sizeMesh)
    f = np.ones(sizeMesh)

    rho_g_RES = []
    rho_l_RES = []
    espilon_RES = []
    x_th_RES = []
    I = []
    conditionnement = []

    for i in range(sizeMesh):
        rho_g_start, rho_l_start, rho_start = getDensity(epsilon[i], P[i]*(10**(-6)), 1)
        rho[i] = rho_start
        rho_g[i] = rho_g_start
        rho_l[i] = rho_l_start
        V_gj[i] = getDriftVelocity(rho_g_start, rho_l_start, g, D_h)
        Vgj_prime[i] = V_gj[i] + (getC0(rho_g_start, rho_l_start) - 1) * U_inlet
        areaMatrix[i] = Area
        #print(f'Friction factor: {getFrictionFactor(getReynoldsNumber(rho[i], U[i], D_h, getMuLiquid(P[i]*(10**(-6)), epsilon[i])), rugo = 0.001, D_h = D_h)}, Reynolds number: {getReynoldsNumber(rho[i], U[i], D_h, getMuLiquid(P[i]*(10**(-6)), epsilon[i]))}, Mu liquid: {getMuLiquid(P[i]*(10**(-6)), epsilon[i])}, Mu gas: {getMuGas(P[i]*(10**(-6)), epsilon[i])}, Phi2phi: {getPhi2Phi(epsilon[i])}')
        areaMatrix_1[i] = getAreas(Area, getPhi2Phi(epsilon[i]), D_h, K_loss, DV, Dz, getFrictionFactor(getReynoldsNumber(rho[i], U[i], D_h, getMuLiquid(P[i]*(10**(-6)), epsilon[i])), rugo = 0.001, D_h = D_h))
        areaMatrix_2[i] = getAreas(Area, - getPhi2Phi(epsilon[i]), D_h, K_loss, DV, Dz, getFrictionFactor(getReynoldsNumber(rho[i], U[i], D_h, getMuLiquid(P[i]*(10**(-6)), epsilon[i])), rugo = 0.001, D_h = D_h))
        Dhfg[i] = getHfg(P[i]*(10**(-6)))
        C0[i] = getC0(rho_g_start, rho_l_start)
        x_th[i] = getThermodynamicQuality(P[i]*(10**(-6)), H[i]*(10**(-3)), U[i], epsilon[i], rho_l[i], rho_g[i], rho[i], Vgj_prime[i], areaMatrix[i])
        f[i] = getFrictionFactor(getReynoldsNumber(rho_start, U_inlet, D_h, getMuLiquid(P[i]*(10**(-6)), epsilon[i])), rugo = 0.001, D_h = D_h)
        print(f'rho_g_start: {rho_g_start}, rho_l_start: {rho_l_start}, rho_start: {rho_start}, epsilon: {epsilon[i]}, x_th: {x_th[i]}, C0: {C0[i]}, V_gj: {V_gj[i]}, Dhfg: {Dhfg[i]}')

    print(f'D_h : {D_h}, \n DV : {DV}, \n rho_g_initial : {rho_g}, \n rho_l_initial : {rho_l}, \n rho_initial : {rho}, \n epsilon_initial : {epsilon}, \n x_th_initial : {x_th}, \n C0_initial : {C0}, \n V_gj_initial : {V_gj}, \n Dhfg_initial : {Dhfg}, \n areaMatrix_initial : {areaMatrix}; \n areaMatrix_old_1 : {areaMatrix_1}; \n areaMatrix_old_2 : {areaMatrix_2}')
    print(f'U_initial : {U}, P_initial : {P}, H_initial : {H}')

    U_old = U
    P_old = P
    H_old = H
    rho_g_old = rho_g
    rho_l_old = rho_l
    rho_old = rho
    V_gj_old = V_gj
    Vgj_prime_old = Vgj_prime
    epsilon_old = epsilon
    areaMatrix = areaMatrix
    areaMatrix_old_1 = areaMatrix_1
    areaMatrix_old_2 = areaMatrix_2
    x_th_old = x_th


    for j in range(N_iterations):

        print(f"\n Begin iteration number: {j}")
        
        VAR_old = createVar(U_old,P_old,H_old)
        rho_old = createVar(rho_old, rho_old, rho_old)
        rho_g_old = createVar(rho_g_old, rho_g_old, rho_g_old)
        rho_l_old = createVar(rho_l_old, rho_l_old, rho_l_old)
        V_gj_old = createVar(V_gj_old, V_gj_old, V_gj_old)
        Vgj_prime = createVar(Vgj_prime, Vgj_prime, Vgj_prime)
        epsilon_old = createVar(epsilon_old, epsilon_old, epsilon_old)
        areaMatrix = createVar(areaMatrix, areaMatrix, areaMatrix)
        areaMatrix_old_1 = createVar(areaMatrix_old_1, areaMatrix_old_1, areaMatrix_old_1)
        areaMatrix_old_2 = createVar(areaMatrix_old_2, areaMatrix_old_2, areaMatrix_old_2)
        Dhfg = createVar(Dhfg, Dhfg, Dhfg)
        C0 = createVar(C0, C0, C0)
        x_th_old = createVar(x_th_old, x_th_old, x_th_old)

        #print(f"Vgj : {V_gj_old}, Vgj_prime: {Vgj_prime}")
        #print(f'Dhfg: {Dhfg}, \n C0: {C0}')

        i = -1
        DI = (1/2) * (VAR_old[i-sizeMesh]*areaMatrix[i] - VAR_old[i-1-sizeMesh]*areaMatrix[i-1]) * ((VAR_old[i-2*sizeMesh]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (VAR_old[i-1-2*sizeMesh]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
        DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
        DM1 = q__ * DV + DI + DI2
        VAR_VFM_Class = FVM(A00 = 1, A01 = 0, Am0 = - rho_old[-2] * VAR_old[sizeMesh-2] * areaMatrix[-2], Am1 = rho_old[-1] * VAR_old[sizeMesh-1] * areaMatrix[-1], D0 = U_inlet, Dm1 = DM1, N_vol = N_vol, H = Height)
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
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(sizeMesh, 
                ci = 0,
                ai = - areaMatrix_old_2[i],
                bi = areaMatrix_old_1[i+1],
                di = - ((rho_old[i+1]- rho_old[i])* g * DV / 2) + DI)
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-sizeMesh,
                ai = - rho_old[i]*VAR_old[i-sizeMesh]*areaMatrix[i],
                bi = rho_old[i+1]*VAR_old[i-sizeMesh]*areaMatrix[i+1])

            elif i > sizeMesh and i < 2*sizeMesh-1:
                DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * areaMatrix[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
                VAR_VFM_Class.set_ADi(i, ci = 0,
                ai = - areaMatrix_old_2[i],
                bi = areaMatrix_old_1[i+1],
                di = - ((rho_old[i+1]- rho_old[i])* g * DV / 2) + DI)
            
                VAR_VFM_Class.fillingOutsideBoundary(i, i-sizeMesh,
                ai = - rho_old[i]*VAR_old[i-sizeMesh]*areaMatrix[i],
                bi = rho_old[i+1]*VAR_old[i+1-sizeMesh]*areaMatrix[i+1])

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
                print(f'VAR_old[i-sizeMesh]: {VAR_old[i-sizeMesh]}, VAR_old[i-1-sizeMesh]: {VAR_old[i-1-sizeMesh]}')
                print(f'VAR_old[i-2*sizeMesh]: {VAR_old[i-2*sizeMesh]}, VAR_old[i-1-2*sizeMesh]: {VAR_old[i-1-2*sizeMesh]}')
                print(f'rho_old[i]: {rho_old[i]}, rho_old[i-1]: {rho_old[i-1]}')
                print(f'epsilon_old[i]: {epsilon_old[i]}, epsilon_old[i-1]: {epsilon_old[i-1]}')
                print(f'V_gj_old[i]: {V_gj_old[i]}, V_gj_old[i-1]: {V_gj_old[i-1]}')
                print(f'areaMatrix[i]: {areaMatrix[i]}, areaMatrix[i-1]: {areaMatrix[i-1]}')
                print(f'areaMatrix_old_1[i]: {areaMatrix_old_1[i]}, areaMatrix_old_1[i-1]: {areaMatrix_old_1[i-1]}')
                print(f'areaMatrix_old_2[i]: {areaMatrix_old_2[i]}, areaMatrix_old_2[i-1]: {areaMatrix_old_2[i-1]}')
                print(f'Dhfg[i]: {Dhfg[i]}, Dhfg[i-1]: {Dhfg[i-1]}')
                print(f'C0[i]: {C0[i]}, C0[i-1]: {C0[i-1]}')
                print(f'x_th_old[i]: {x_th_old[i]}, x_th_old[i-1]: {x_th_old[i-1]}')
                DI = (1/2) * (VAR_old[i-sizeMesh]*areaMatrix[i] - VAR_old[i-1-sizeMesh]*areaMatrix[i-1]) * ((VAR_old[i-2*sizeMesh]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (VAR_old[i-1-2*sizeMesh]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
                DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*areaMatrix[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
                print(f'DI: {DI}, DI2: {DI2}, q__: {q__}, DV: {DV}, q__DV: {q__ * DV}')
                VAR_VFM_Class.set_ADi(i, ci =  - rho_old[i-1] * VAR_old[i-1-2*sizeMesh] * areaMatrix[i-1],
                ai = rho_old[i] * VAR_old[i-2*sizeMesh] * areaMatrix[i],
                bi = 0,
                di =  q__ * DV + DI + DI2)

        #print(f'VAR_VFM_Class.A: {VAR_VFM_Class.A}, VAR_VFM_Class.D: {VAR_VFM_Class.D}')
        conditionnement.append(np.linalg.cond(VAR_VFM_Class.A))
        VAR = VAR_VFM_Class.resoudre_h()
        U, P, H = splitVar(VAR)
        print(f'U: {U} m/s, P: {P} Pa, H: {H} J/kg')

        rho_old, rho_old0, rho_old0 = splitVar(rho_old)
        rho_g_old, rho_g_old, rho_g_old = splitVar(rho_g_old)
        rho_l_old, rho_l_old, rho_l_old = splitVar(rho_l_old)
        V_gj_old, V_gj_old, V_gj_old = splitVar(V_gj_old)
        Vgj_prime, Vgj_prime, Vgj_prime = splitVar(Vgj_prime)
        epsilon_old, epsilon_old, epsilon_old = splitVar(epsilon_old)
        areaMatrix, areaMatrix, areaMatrix = splitVar(areaMatrix)
        areaMatrix_old_1, areaMatrix_old_1, areaMatrix_old_1= splitVar(areaMatrix_old_1)
        areaMatrix_old_2, areaMatrix_old_2, areaMatrix_old_2 = splitVar(areaMatrix_old_2)
        Dhfg, Dhfg, Dhfg = splitVar(Dhfg)
        C0, C0, C0 = splitVar(C0)
        x_th_old, x_th_old, x_th_old = splitVar(x_th_old)

        print(f"Avant update fields: rho_g_old: {rho_g_old}, rho_l_old: {rho_l_old}, rho_old: {rho_old}")

        #Update fields
        for i in range(sizeMesh):
            #print(f'Before getParameters: rho_g_old: {rho_g_old}, rho_l_old: {rho_l_old}, rho_old: {rho_old}, epsilon_old: {epsilon_old}, x_th: {x_th}, C0: {C0}, V_gj_old: {V_gj_old}, Dhfg: {Dhfg}')
            rho_g[i], rho_l[i], rho[i], epsilon[i], x_th[i], C0[i], V_gj[i], Dhfg[i] = get_parameters(P[i]*(10**(-6)), H[i]*(10**(-3)), rho_l_old[i], rho_g_old[i], rho_old[i], epsilon_old[i], D_h, g, U[i], areaMatrix[i])
            #print(f'After getParameters: rho_g: {rho_g}, rho_l: {rho_l}, rho: {rho}, epsilon: {epsilon}, x_th: {x_th}, C0: {C0}, V_gj: {V_gj}, Dhfg: {Dhfg}')
            areaMatrix_old_1[i] = getAreas(Area, getPhi2Phi(epsilon[i]), D_h, K_loss, DV, Dz, getFrictionFactor(getReynoldsNumber(rho[i], U[i], D_h, getMuLiquid(P[i]*(10**(-6)), epsilon[i])), rugo = 0.001, D_h = D_h))
            areaMatrix_old_2[i] = getAreas(Area, getPhi2Phi(epsilon[i]), D_h, K_loss, DV, Dz, getFrictionFactor(getReynoldsNumber(rho[i], U[i], D_h, getMuLiquid(P[i]*(10**(-6)), epsilon[i])), rugo = 0.001, D_h = D_h))
        
        print(f"Après update fields: x_th: {x_th}, rho_g : {rho_g}, rho_l: {rho_l}, rho: {rho}, epsilon: {epsilon}, C0: {C0}, V_gj: {V_gj}, Dhfg: {Dhfg}")

        #Sous relaxation
        rho_g_relaxed = sousRelaxation(rho_g, rho_g_old, sousRelaxFactor)
        rho_l_relaxed = sousRelaxation(rho_l, rho_l_old, sousRelaxFactor)
        rho_relaxed = sousRelaxation(rho, rho_old, sousRelaxFactor)
        epsilon_relaxed = sousRelaxation(epsilon, epsilon_old, sousRelaxFactor)
        x_th_relaxed = sousRelaxation(x_th, x_th_old, sousRelaxFactor)
        Vgj_relaxed = sousRelaxation(V_gj, V_gj_old, sousRelaxFactor)
        C0_relaxed = sousRelaxation(C0, C0, sousRelaxFactor)
        Vgj_prime_relaxed = sousRelaxation(Vgj_prime, Vgj_prime_old, sousRelaxFactor)
        #H_relaxed = sousRelaxation(H, H_old, sousRelaxFactor)
        
        print(f"rho_g_relaxed: {rho_g_relaxed}, \n rho_g {rho_g}, \n rho_g_old: {rho_g_old}")
        print(f"rho_l_relaxed: {rho_l_relaxed}, \n rho_l {rho_l}, \n rho_l_old: {rho_l_old}")
        print(f"rho_relaxed: {rho_relaxed}, \n rho {rho}, \n rho_old: {rho_old}")
        
        U_residual = np.linalg.norm(np.array(U) - np.array(U_old))
        P_residual = np.linalg.norm(np.array(P) - np.array(P_old))
        H_residual = np.linalg.norm(np.array(H) - np.array(H_old))

        rho_g_residual = np.linalg.norm(np.array(rho_g_relaxed) - np.array(rho_g_old))
        rho_l_residual = np.linalg.norm(np.array(rho_l_relaxed) - np.array(rho_l_old))
        epsilon_residual = np.linalg.norm(np.array(epsilon_relaxed) - np.array(epsilon_old))
        x_th_residual = np.linalg.norm(np.array(x_th_relaxed) - np.array(x_th_old))
        Vgj_residual = np.linalg.norm(np.array(Vgj_relaxed) - np.array(V_gj_old))
        C0_residual = np.linalg.norm(np.array(C0_relaxed) - np.array(C0))
        Vgj_prime_residual = np.linalg.norm(np.array(Vgj_prime_relaxed) - np.array(Vgj_prime_old))

        rho_g_RES.append(rho_g_residual)
        rho_l_RES.append(rho_l_residual)
        espilon_RES.append(epsilon_residual)
        x_th_RES.append(x_th_residual)
        I.append(j)

        if (rho_g_residual < eps) and (rho_l_residual < eps) and (epsilon_residual < eps) and (x_th_residual < eps):
            print(f"Iteration number: {j}, rho_g_residual: {rho_g_residual}, rho_l_residual: {rho_l_residual}, epsilon_residual: {epsilon_residual}, x_th_residual: {x_th_residual}")
            print(f"Convergence reached at iteration {j}")
            print(f'U: {U}, \n P: {P},\n H: {H}')
            print(f'rho_g: {rho_g}, \n rho_l: {rho_l}, \n rho: {rho},\n epsilon: {epsilon},\n x_th: {x_th},\n C0: {C0},\n V_gj: {V_gj},\n Dhfg: {Dhfg}')
            print(f'rho_g_old: {rho_g_old}, rho_l_old: {rho_l_old}, rho_old: {rho_old}, epsilon_old: {epsilon_old}, x_th: {x_th}, C0: {C0}, V_gj_old: {V_gj_old}, Dhfg: {Dhfg}')
            break

        elif j == N_iterations - 1:
            print(f'rho_old: {rho_old}, Area : {Area})')
            print("The system did not converge")
            break
            
        else:
            rho_g_old = rho_g_relaxed
            rho_l_old = rho_l_relaxed
            rho_old = rho_relaxed
            epsilon_old = epsilon_relaxed
            x_th_old = x_th_relaxed
            V_gj_old = Vgj_relaxed
            C0 = C0_relaxed
            Vgj_prime_old = Vgj_prime_relaxed
            H_old = H
            U_old = U
            P_old = P
            print(f"Itération number: {j}, rho_g_residual: {rho_g_residual}, rho_l_residual: {rho_l_residual}, epsilon_residual: {epsilon_residual}, x_th_residual: {x_th_residual}")
            print(f"Convergence not reached yet at iteration {j}")

    T=[]
    Hlsat = []
    Hgsat = []
    for i in range(sizeMesh):
        T.append(getTemperature(P[i]*(10**(-6)), H[i]*(10**(-3)), epsilon[i]))
        P[i] = P[i]*(10**(-6))
        H[i] = H[i]*(10**(-3))
        Hlsat.append(getHlsat(P[i]))
        Hgsat.append(getHgsat(P[i]))

    print(f'Final values: zList: {zList} m, U: {U}\n P: {P}\n H: {H}\n T: {T}, \n epsilon: {epsilon}, ')
    print(f'Conditionnement: {conditionnement}')
    print(f'minimum conditionnement: {min(conditionnement)}, maximum conditionnement {max(conditionnement)}')
    Twater_THM = [502.15841096, 506.45412982, 510.72058767, 514.95638224, 519.16004561, 523.33003889, 527.46474596, 531.56246619, 535.62140573, 539.63966744, 543.61523907, 547.54597958, 551.42960328, 555.2636616,  559.04552227, 562.77234577, 566.44105933, 570.0483291 , 573.59053173, 577.06372668]   

    Tsat = []
    Psat = []
    Hsat = []
    for i in range(sizeMesh):
        Tsat.append(IAPWS97(P = P[i], x = 0.5).T)
        Psat.append(IAPWS97(T = T[i], x = 0.5).P)

    print(T)
    #plt.plot(zList, Twater_THM, label="Twater_THM")
    fig1 = plt.figure()
    plt.plot(zList, T, label="Twater_DFM")
    plt.plot(zList, Tsat, label="Tsat")
    plt.xlabel("z [m]")
    plt.ylabel("T [K]")
    plt.legend()
    #plt.savefig("Twater_THM_DFM.png")

    fig2 = plt.figure()
    plt.plot(zList, U, label="U")
    plt.xlabel("z [m]")
    plt.ylabel("U [m/s]")
    plt.legend()

    fig3 = plt.figure()
    plt.plot(zList, P, label="P")
    plt.xlabel("z [m]")
    plt.ylabel("P [Pa]")
    plt.ylim((np.min(P), np.max(P)))
    plt.legend()

    fig1 = plt.figure()
    plt.plot(zList, H, label="H")
    plt.plot(zList, Hlsat, label="Hlsat")
    plt.plot(zList, Hgsat, label="Hgsat")
    plt.xlabel("z [m]")
    plt.ylabel("H [J/kg]")
    plt.legend()

    fig5 = plt.figure()
    plt.plot(zList, epsilon, label="epsilon")
    plt.xlabel("z [m]")
    plt.ylabel("epsilon")
    plt.legend()

    fig23 = plt.figure()
    plt.plot(zList, x_th, label="x_th")
    plt.xlabel("z [m]")
    plt.ylabel("x_th")
    plt.legend()

    fig6 = plt.figure()
    plt.plot(zList, rho, label="density")
    plt.xlabel("z [m]")
    plt.ylabel("density [kg/m3]")
    plt.legend()

    fig7 = plt.figure()
    plt.plot(I, rho_g_RES, label="rho_g_residual")
    plt.plot(I, rho_l_RES, label="rho_l_residual")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.legend()

    fig8 = plt.figure()
    plt.plot(I, espilon_RES, label="epsilon_residual")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.legend()

    fig9 = plt.figure()
    plt.plot(I, x_th_RES, label="x_th_residual")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.legend()

    fig10= plt.figure()
    plt.plot(I, conditionnement, label="conditionnement")
    plt.xlabel("Iteration")
    plt.ylabel("conditionnement")
    plt.legend()


    plt.show()


""" fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(zList, Twater_THM, label="Twater_THM")
axs[0, 0].plot(zList, T, label="Twater_DFM")
axs[0, 0].set_title('Comparison of the temperature profiles')
axs[0, 1].plot(zList, U, 'tab:orange')
axs[0, 1].set_title('Velocity (m/s)')
axs[1, 0].plot(zList, P, 'tab:green')
axs[1, 0].set_title('Pressure (Pa)')
axs[1, 1].plot(zList, H, 'tab:red')
axs[1, 1].set_title('H')

for ax in axs.flat:
    ax.set(xlabel='z (m)')

axs[0, 0].set(ylabel='T (K)')
axs[0, 1].set(ylabel='U (m/s)')
axs[1, 0].set(ylabel='P (Pa)')
axs[1, 1].set(ylabel='H (J/kg)')

plt.show() """

main()

#cProfile.run('main()')