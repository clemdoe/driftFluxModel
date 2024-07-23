import numpy as np
from function import *
from iapws import IAPWS97

print('Beginning of the script')

# Equation resolution parameters
eps = 10**(-3)
N_iterations = 1000

# Constant of the problem
N = 4 #Number of volumes for the discretization using the FVM class
cote = 0.03245
Poro = (0.3860632848 + 0.4151712309 + 0.4464738237 + 0.3860632848) / 4
FlowArea = cote ** 2 * Poro #m2
Height = 4.288 #m
g = 9.81 #gravity m/s2
K_loss = 0 #loss coefficient

fuelRadius = 5.6*10**(-3) #External radius of the fuel m
cladRadius = 6.52*10**(-3) #External radius of the clad m
waterGap = 0.5*10**(-3) #Gap between the clad and the water m
waterRadius =  cladRadius + waterGap #External radius of the water m
#Boundary conditions
P_outlet = 16602658.5 #Pa
U_inlet = 6.70 #m/s
T_inlet = 587.8 #K
P_inlet = 14.9 * (10**6) #Pa
h_inlet = IAPWS97(T = T_inlet, P = P_inlet * 10**(-6)).h * 1000 #J/kg

#Calculated
DV = (Height/N) * FlowArea #Volume of the control volume m3
Dz = Height/N #Height of the control volume m
D_h = (0.0079178195 + 0.0094095898 + 0.0117778432 + 0.0079178195) / 4  #Hydraulic diameter m2
A = np.ones(N)#*FlowArea

#Power input

#Shift in the heating
startHeating = 0 #m
endHeating = Height #m
Q = [500000000 for i in range(N)]
print(f'len Q:', len(Q))
q__fluid = np.zeros(N) #W/m
q__ = np.zeros(N) #W/m3

for i in range(N):
    if i*Dz < startHeating:
        q__[i] = 0
    elif i*Dz > startHeating and i*Dz < endHeating:
        q__fluid[i] = np.pi * fuelRadius**2 * Q[int(i- startHeating/Dz)] #linear heat generation rate W/m
        q__[i] = q__fluid[i] / FlowArea #W/m3
    else:
        q__[i] = 0
    
print(f'q__: {q__}')

print(A)

def setBoundaryConditions(X, u_inlet, u_outlet, p_outlet, h_inlet):

    #Velocity
    X[0] = u_inlet
    X[1:N] = [u_inlet -1  for i in range(N-1)]

    #Pressure
    X[N:2*N-1] = [p_outlet -1  for i in range(N-1)]
    X[2*N-1] = p_outlet

    #Enthalpy
    X[2*N] = h_inlet
    X[2*N+1:3*N] = [h_inlet - 1 for i in range(N-1)]

    return X

def F(X, X_old, parametersCurF, parametersOldF, q):

    q__ = q

    u = X[:N]
    P = X[N:2*N]
    h = X[2*N:]
    print(f'len h : {len(h)}')
    u_ = X_old[:N]
    P_old = X_old[N:2*N]
    h_old = X_old[2*N:]

    density = parametersCurF[:,0]

    rho_old, rho_l_old, rho_g_old, epsilon_old, V_gj_old, C0_old, x_th_old, Dhfg = parametersOldF[:,0], parametersOldF[:,1], parametersOldF[:,2], parametersOldF[:,3], parametersOldF[:,4], parametersOldF[:,5], parametersOldF[:,6], parametersOldF[:,7]

    F = np.zeros(3 * N)
    F[0] = 0
    F[2*N-1] = 0
    F[2*N] = 0
    # Mass concervation equation
    for i in range(1, N):
        print(f'{i}, density[i]: {density[i]}, u[i]: {u[i]}, density[i-1]: {density[i-1]}, u[i-1]: {u[i-1]}')
        F[i] = density[i] * u[i] - density[i-1] * u[i-1]

    # Momentum concervation equation
    for i in range(0, N-1):
        DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * A[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * A[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
        secondMember = - ((rho_old[i+1]- rho_old[i])* g * DV / 2) + DI
        print(f'secondMember: {secondMember}, DI: {DI}')
        F[N+i] = (density[i+1] * A[i+1] * u[i+1] * u_[i+1] - density[i] * A[i] * u[i] * u_[i] ) + A[i+1] * P[i+1] - A[i] * P[i] - secondMember

    # Energy concervation equation
    for i in range(1, N):
        DI = (1/2) * (P_old[i]*A[i] - P_old[i-1]*A[i-1]) * ((u_[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (u_[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
        DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*A[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*A[i-1]/rho_old[i-1])
        secondMember =  q__[i] * DV + DI + DI2
        print(f'secondMember: {secondMember}, DI: {DI}, DI2: {DI2}')
        F[2*N+i] = density[i] * u_[i] * A[i] * h[i] - density[i-1] * u_[i-1] * A[i-1] * h[i-1] - secondMember
    
    return F

def jacobian(A, X_, parametersCurF, parametersOldF):

    density = parametersOldF[:,0]
    u_ = X_[:N]

    J = np.zeros((3*N,3* N))
    for i in range(0, N):
        if i == 0:
            #Inside the velocity submatrix
            J[i, i] = 1

            #Inside the pressure submatrix
            J[i+N, i+N] = - A[i]
            J[i+N, i+N+1] = A[i+1]

            J[i+N, i] = - density[i] * u_[i] * A[i]
            J[i+N, i+1] =  density[i] * u_[i] * A[i]

            #Inside the enthalpy submatrix
            J[i+2*N, i+2*N] = 1

        elif i == N-1:
            #Inside the velocity submatrix
            J[i, i] = density[i]*A[i]
            J[i, i-1] = - density[i-1]*A[i-1]

            #Inside the pressure submatrix
            J[i+N, i+N] = 1

            #Inside the enthalpy submatrix
            J[i+2*N, i+2*N-1] = - density[i-1] * u_[i-1] * A[i-1]
            J[i+2*N, i+2*N] =  density[i] * u_[i] * A[i]

        else:
            #Inside the velocity submatrix
            J[i, i] = density[i]*A[i]
            J[i, i-1] = - density[i-1]*A[i-1]

            #Inside the pressure submatrix
            J[i+N, i+N] = - A[i]
            J[i+N, i+N+1] = A[i+1]

            J[i+N, i] =  - density[i] * u_[i] * A[i]
            J[i+N, i+1] =   density[i] * u_[i] * A[i]

            #Inside the enthalpy submatrix
            J[i+2*N, i+2*N-1] = - density[i-1] * u_[i-1] * A[i-1]
            J[i+2*N, i+2*N] = density[i] * u_[i] * A[i]

    return J

def newton_raphson(X, X_, A, parametersCurF, parametersOldF, q, tol=1e-4, max_iter=10): #Ok

    for k in range(max_iter):
        print(f'Iteration {k}')
        F_val = F(X, X_, parametersCurF, parametersOldF, q)
        print(f'F_val: {F_val}')
        if np.linalg.norm(F_val) < tol:
            print('Converged in', k, 'iterations')
            print(f'Final X:', X)
            return X

        J = jacobian(A, X_, parametersCurF, parametersOldF)
        print('J:', J, 'F_val:', F_val)
        delta_U = np.linalg.solve(J, -F_val)
        X_ = X
        print(f'X_ : {X_}')
        X += delta_U
        print(f'X_ : {X_}')
        

        if np.linalg.norm(delta_U) < tol:
            print('Converged in', k, 'iterations')
            print(f'Final X:', X)
            return X
    
    print('Did not converge')

X_old = np.ones(3*N)
X = setBoundaryConditions(X_old, U_inlet, 0, P_outlet, h_inlet)
print(f'X: {X}')
print(f'X_old: {X_old}')

#rho_l_old, rho_g_old, epsilon_old, V_gj_old, P_old, Dhfg 
parametersOld= np.zeros((N,8))
parametersOld[:,0] = [1000, 1000, 1000, 1000]
print(f'parametersOld: {parametersOld}')

parametersCur = np.zeros((N,8))
parametersCur[:,0] = [1000, 1000, 1000, 1000]

newton_raphson(X, X_old, A, parametersCur, parametersOld, q__)


print('End of the script')