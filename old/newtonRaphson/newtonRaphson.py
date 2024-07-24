import numpy as np
from function import *
from iapws import IAPWS97
import matplotlib.pyplot as plt

print('Beginning of the script')

# Equation resolution parameters
eps = 10**(-3)
N_iterations = 1000

# Constant of the problem
N = 10 #Number of volumes for the discretization using the FVM class
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
zlist = [i*Dz for i in range(N)]

#Power input

#Shift in the heating
startHeating = 0 #m
endHeating = Height #m
Q = [700000000 for i in range(N)] #W/m3
q__fluid = np.zeros(N) #W/m
q__ = np.zeros(N) #W/m3

for i in range(N):
    if i*Dz < startHeating:
        q__[i] = 0
    elif i*Dz >= startHeating and i*Dz <= endHeating:
        q__fluid[i] = np.pi * fuelRadius**2 * Q[int(i- startHeating/Dz)] #linear heat generation rate W/m
        q__[i] = q__fluid[i] / FlowArea #W/m3
    else:
        q__[i] = 0
    
print(f'q__: {q__}')

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

def F(X, X_old, parametersOldF, q):

    q__ = q

    u = X[:N]
    P = X[N:2*N]
    h = X[2*N:]
    u_ = X_old[:N]
    P_old = X_old[N:2*N]
    h_old = X_old[2*N:]

    rho_old, rho_l_old, rho_g_old, epsilon_old, V_gj_old, C0_old, x_th_old, Dhfg = parametersOldF[:,0], parametersOldF[:,1], parametersOldF[:,2], parametersOldF[:,3], parametersOldF[:,4], parametersOldF[:,5], parametersOldF[:,6], parametersOldF[:,7]

    F = np.zeros(3 * N)
    F[0] = 0
    F[2*N-1] = 0
    F[2*N] = 0
    # Mass concervation equation
    for i in range(1, N):
        F[i] = rho_old[i] * u[i] - rho_old[i-1] * u[i-1]

    # Momentum concervation equation
    for i in range(0, N-1):
        DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * A[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * A[i] )/ ((1 - epsilon_old[i])*rho_old[i]) )     
        secondMember = - ((rho_old[i+1]- rho_old[i])* g * DV / 2) + DI
        F[N+i] = (rho_old[i+1] * A[i+1] * u[i+1] * u_[i+1] - rho_old[i] * A[i] * u[i] * u_[i] ) + A[i+1] * P[i+1] - A[i] * P[i] - secondMember

    # Energy concervation equation
    for i in range(1, N):
        DI = (1/2) * (P_old[i]*A[i] - P_old[i-1]*A[i-1]) * ((u_[i]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (u_[i-1]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
        DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*A[i]/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*A[i-1]/rho_old[i-1])
        secondMember =  q__[i] * DV + DI + DI2
        F[2*N+i] = rho_old[i] * u_[i] * A[i] * h[i] - rho_old[i-1] * u_[i-1] * A[i-1] * h[i-1] - secondMember
    
    return F

def jacobian(A, X_, parametersOldF):

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

def newton_raphson(X, X_, A, parametersOldF, q, tol=1e-4, max_iter=10): #Ok

    for k in range(max_iter):
        #print(f'Iteration {k}')
        F_val = F(X, X_, parametersOldF, q)
        print(f'F_val: {F_val}, tol: {tol}')
        if np.linalg.norm(F_val) < tol:
            #print('Converged in', k, 'iterations')
            #print(f'Final X:', X)
            return X

        J = jacobian(A, X_, parametersOldF)
        #print('J:', J, 'F_val:', F_val)
        delta_U = np.linalg.solve(J, -F_val)
        X_ = X
        #print(f'X_ : {X_}')
        X += delta_U
        

        if np.linalg.norm(delta_U) < tol:
            #print('Converged in', k, 'iterations in Newton Raphson')
            print(f'Final X:', X)
            return X
    
    print('Did not converge')

X_old = np.ones(3*N)
X = setBoundaryConditions(X_old, U_inlet, 0, P_outlet, h_inlet)
print(f'X: {X}')
print(f'X_old: {X_old}')

#rho_l_old, rho_g_old, epsilon_old, V_gj_old, P_old, Dhfg 

parametersOld= np.zeros((N,8))
print(f'parametersOld: {parametersOld}')

parametersCur = np.zeros((N,8))
parametersOld[:,0] = [1000 for i in range(N)] #rho
parametersOld[:,1] = [916.8 for i in range(N)]  #rho_l
parametersOld[:,2] = [1000 for i in range(N)] #rho_g
parametersOld[:,3] = [0.001 for i in range(N)] #epsilon
parametersOld[:,6] = [0.001 for i in range(N)] #epsilon

u = X[:N]
P = X[N:2*N]
h = X[2*N:]

for k in range(1000):
    for j in range(0, N):
        #Calculate the density of the mixture
        parametersCur[j,2], parametersCur[j,1], parametersCur[j,0] = getDensity(P[j], h[j], parametersOld[j,3])
        #Calculate the drift velocity of the mixture
        parametersCur[j,4] = getV_gj(parametersOld[j,1], parametersOld[j,2], D_h, g)
        #Calculate the constant C0 called the distribution parameter
        parametersCur[j,5] = getC0(parametersOld[j,2], parametersOld[j,1])
        #Calculate the void fraction of the mixture
        parametersCur[j,3] = VoidFraction(parametersOld, j)
        #Calculate the thermodynamic quality of the mixture
        parametersCur[j,6] = Quality(parametersOld, j)
        #Calculate the heat of vaporization
        parametersCur[j,7] = getHfg(getTemperature(P[j], h[j]))

    if np.linalg.norm(parametersCur[:,3] - parametersOld[:,3]) < eps and np.linalg.norm(parametersCur[:,2] - parametersOld[:,2]) < eps and np.linalg.norm(parametersCur[:,1] - parametersOld[:,1]) < eps and np.linalg.norm(parametersCur[:,6] - parametersOld[:,6]) < eps:
        break
    else:
        parametersOld = parametersCur.copy()

#rho_old, rho_l_old, rho_g_old, epsilon_old, V_gj_old, C0_old, x_th_old, Dhfg


parametersCur = parametersOld.copy()    
print(f'First set of parameters: {parametersCur}')

for i in range(N_iterations):
    print(f'Iteration {i}')

    X_new = newton_raphson(X, X_old, A, parametersOld, q__)

    u = X_new[:N]
    P = X_new[N:2*N]
    h = X_new[2*N:]
    
    #Calculate the parameters of the mixture
    for k in range(1000):
        for j in range(0, N):
            #Calculate the density of the mixture
            parametersCur[j,2], parametersCur[j,1], parametersCur[j,0] = getDensity(P[j], h[j], parametersOld[j,3])
            #Calculate the drift velocity of the mixture
            parametersCur[j,4] = getV_gj(parametersCur[j,1], parametersOld[j,2], D_h, g)
            #Calculate the constant C0 called the distribution parameter
            parametersCur[j,5] = getC0(parametersCur[j,2], parametersOld[j,1])
            #Calculate the void fraction of the mixture
            parametersCur[j,3] = VoidFraction(parametersOld, j)
            #Calculate the thermodynamic quality of the mixture
            parametersCur[j,6] = Quality(parametersOld, j)
            #Calculate the heat of vaporization
            parametersCur[j,7] = getHfg(getTemperature(P[j], h[j]))

        if np.linalg.norm(parametersCur[:,3] - parametersOld[:,3]) < eps and np.linalg.norm(parametersCur[:,2] - parametersOld[:,2]) < eps and np.linalg.norm(parametersCur[:,1] - parametersOld[:,1]) < eps and np.linalg.norm(parametersCur[:,6] - parametersOld[:,6]) < eps:
            break
        else:
            parametersOld = parametersCur.copy()
    
    if np.linalg.norm(X_new - X) < eps:
        print(f'Converged in {k} iterations')
        break

    elif i == N_iterations:
        print('Did not converge')

    else:
        X_old = X.copy()
        X = X_new.copy()

u = X[:N]
P = X[N:2*N]
h = X[2*N:]
epsilon = parametersCur[:,3]
rho = parametersCur[:,0]
x_th = parametersCur[:,6]
print(f'Final u: {u}')
print(f'Final P: {P}')
print(f'Final h: {h}')
print(f'Final epsilon: {epsilon}')
print(f'Final rho: {rho}')
print(f'Final x_th: {x_th}')


T=[]
for i in range(N):
    T.append(getTemperature(P[i], h[i]) - 273.15)
    h[i] = h[i] / 1000
    P[i] = P[i] / 10**6

plt.plot(zlist, T, label = 'Temperature')
plt.xlabel('Height [m]')
plt.ylabel('Temperature [°C]')
plt.yscale('linear')
plt.legend()
plt.show()

plt.plot(zlist, u, label = 'Velocity')
plt.xlabel('Height [m]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.show()

plt.plot(zlist, P, label = 'Pressure')
plt.xlabel('Height [m]')
plt.ylabel('Pressure [MPa]')
plt.yscale('linear')
plt.legend()
plt.show()

""" plt.plot(zlist, epsilon, label = 'Void fraction')
plt.xlabel('Height [m]')
plt.ylabel('Void fraction')
plt.legend()
plt.show()
 """
plt.plot(zlist, h, label = 'Enthalpy')
plt.xlabel('Height [m]')
plt.ylabel('Enthalpy [kJ/kg]')
plt.yscale('linear')
plt.legend()
plt.show()

plt.plot(zlist, rho, label = 'Density')
plt.xlabel('Height [m]')
plt.ylabel('Density [kg/m3]')
plt.ylim(min(rho)-1, max(rho)+1)
plt.legend()
plt.show()


print('End of the script')