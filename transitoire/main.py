import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from VFClass import VF_1D

uInlet = 4.68292412
pOutlet = 14739394.95
T_inlet = 602.75 #K
P_inlet = 14.9 * (10**6) #Pa
hInlet = IAPWS97(T = T_inlet, P = P_inlet * 10**(-6)).h * 1000 #J/kg
    
rho = 900
sizeMesh = 5
g = 9.81

cote = 0.0157
Poro = 0.5655077285
flowArea = cote ** 2 * Poro #m2

#Geometry parameters
height = 1.655 #m
fuelRadius = 5.6*10**(-3) #External radius of the fuel m
cladRadius = 6.52*10**(-3) #External radius of the clad m
waterGap = 0.5*10**(-3) #Gap between the clad and the water m
waterRadius =  cladRadius + waterGap #External radius of the water m

DV = (height/sizeMesh) * flowArea #Volume of the control volume m3
D_h = 0.0078395462 #Hydraulic diameter m2
zList = np.linspace(0, height, sizeMesh)

massConcerv = VF_1D(L = height, N_vol = 3 * sizeMesh)
massConcerv.set_transitoire(t_tot = 0.1, Xini = [uInlet] * sizeMesh + [pOutlet]* sizeMesh + [hInlet] * sizeMesh, dt = 0.01)

#Heating parameters
Q = 500000000 #volumetric heat generation rate W/m3
q__fluid = np.pi * fuelRadius**2 * Q #linear heat generation rate W/m
q__ = q__fluid / flowArea #W/m3

rho = np.zeros((massConcerv.N_temps, massConcerv.N_vol))
rho[0] = [900] * massConcerv.N_vol
rho[1] = [900] * massConcerv.N_vol
VAR = np.zeros((massConcerv.N_temps, massConcerv.N_vol))
VAR[0] = [uInlet] * sizeMesh + [pOutlet]* sizeMesh + [hInlet] * sizeMesh

def getDensity(temperature, pressure):
    return IAPWS97(P = pressure * 10**(-6), T = temperature).rho

def getTemperature(pressure, enthalpy):
    return IAPWS97(P = pressure * 10**(-6), h = enthalpy * (10**(-3))).T

for t in range(1, massConcerv.N_temps):
    for i in range(1, massConcerv.N_vol - 1):
        if i < sizeMesh:
            massConcerv.set_ADi(i, 
                    ci = - rho[t][i-1] * flowArea, 
                    ai = rho[t][i] * flowArea, 
                    bi = 0, 
                    di = flowArea * (rho[t][i] - rho[t-1][i]) * massConcerv.dx / massConcerv.dt)

        if i == sizeMesh:
            #DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * flowArea )/ ((1 - epsilon_old[i])*rho_old[i]) )     
            massConcerv.set_ADi(sizeMesh, 
                ci = 0,
                ai = - flowArea,
                bi = flowArea,
                di = - ((rho[t][i+1]- rho[t][i])* g * DV / 2 ) + flowArea * (rho[t][i]*VAR[t-1][i - sizeMesh] - rho[t-1][i] * VAR[t-1][i]) * massConcerv.dx / massConcerv.dt) #+ DI)
            
            massConcerv.fillingOutsideBoundary(i, i-sizeMesh,
                ai = - rho[t][i]*VAR[t-1][i-sizeMesh]*flowArea,
                bi = rho[t][i+1]*VAR[t-1][i-sizeMesh +1]*flowArea)
            
        elif i > sizeMesh and i < 2*sizeMesh-1:
            #DI = -((epsilon_old[i+1] * rho_g_old[i+1] * rho_l_old[i+1] * V_gj_old[i+1]**2 * areaMatrix[i+1] )/ ((1 - epsilon_old[i+1])*rho_old[i+1]) )  + ((epsilon_old[i] * rho_g_old[i] * rho_l_old[i] * V_gj_old[i]**2 * flowArea )/ ((1 - epsilon_old[i])*rho_old[i]) )     
            massConcerv.set_ADi(i, ci = 0,
                ai = - flowArea,
                bi = flowArea,
                di = - ((rho[t][i+1]- rho[t][i])* g * DV / 2) + flowArea * (rho[t][i]*VAR[t-1][i - sizeMesh] - rho[t-1][i] * VAR[t-1][i]) * massConcerv.dx / massConcerv.dt) #+ DI)
            
            massConcerv.fillingOutsideBoundary(i, i-sizeMesh,
                ai = - rho[t][i]*VAR[t-1][i-sizeMesh]*flowArea,
                bi = rho[t][i+1]*VAR[t-1][i+1-sizeMesh]*flowArea)
        
        #Inside the enthalpy submatrix
        elif i == 2*sizeMesh:
            massConcerv.set_ADi(2*sizeMesh, 
                ci = 0,
                ai = 1,
                bi = 0,
                di =  hInlet)
            
        elif i > 2*sizeMesh and i < 3*sizeMesh:
            #DI = (1/2) * (VAR[i-sizeMesh]*flowArea - VAR[i-1-sizeMesh]*areaMatrix[i-1]) * ((VAR[i-2*sizeMesh]+ ((epsilon_old[i] * (rho_l_old[i] - rho_g_old[i]) * V_gj_old[i])/ rho_old[i]))+ (VAR[i-1-2*sizeMesh]+ ((epsilon_old[i-1] * (rho_l_old[i-1] - rho_g_old[i-1]) * V_gj_old[i-1])/ rho_old[i-1]) ) )
            #DI2 = - (epsilon_old[i]*rho_l_old[i]*rho_g_old[i]*Dhfg[i]*V_gj_old[i]*flowArea/rho_old[i]) + (epsilon_old[i-1]*rho_l_old[i-1]*rho_g_old[i-1]*Dhfg[i-1]*V_gj_old[i-1]*areaMatrix[i-1]/rho_old[i-1])
            print(f'rho: {rho[t][i-1]}, VAR: {VAR[t-1][i-1-2*sizeMesh]}, flowArea: {flowArea}, rho: {rho[t][i-1]}, VAR: {VAR[t-1][i-2*sizeMesh]}, flowArea: {flowArea}, q__: {q__}, DV: {DV}')
            massConcerv.set_ADi(i, ci =  - rho[t][i-1] * VAR[t-1][i-1-2*sizeMesh] * flowArea,
            ai = rho[t][i] * VAR[t-1][i-2*sizeMesh] * flowArea,
            bi = 0,
            di = q__ * DV )#+ DI + DI2)
            
    A0, Am1 = np.zeros(massConcerv.N_vol), np.zeros(massConcerv.N_vol)
    A0[:2] = [ 1 , 0]
    Am1[-2:] = [  - rho[t][i-1] * VAR[t-1][i-1-2*sizeMesh] * flowArea , rho[t][i] * VAR[t-1][i-2*sizeMesh] * flowArea]
    massConcerv.set_CL(A0, Am1, uInlet, 0)
    print(massConcerv.A)
    print(massConcerv.D)
    massConcerv.X[t] = massConcerv.resoudre_X()

    if t <= massConcerv.N_temps - 1:
        for i in range(sizeMesh):
            pressure = massConcerv.X[t-1][i + sizeMesh]
            enthalpy = massConcerv.X[t-1][i + 2*sizeMesh]
            temperature = getTemperature(pressure, enthalpy)
            rho[t+1][i] = getDensity(temperature, pressure)

print(massConcerv.X)
    
def mergeVAR(U, P):
    return np.hstack((U, P))

def splitVAR(X):
    return X[:, :sizeMesh], X[:, sizeMesh:]

U, P = splitVAR(massConcerv.X)
print(U[:, -1])

fig1 = plt.figure()
plt.plot(zList, U[-1])
plt.xlabel('z')
plt.ylabel('Vitesse')

fig2 = plt.figure()
plt.plot(massConcerv.tList, U[:,-1])
plt.xlabel('t')
plt.ylabel('Vitesse')

fig3 = plt.figure()
plt.plot(zList, P[-1])
plt.xlabel('z')
plt.ylabel('Pression')


fig4 = plt.figure()
plt.plot(massConcerv.tList, P[:,-1])
plt.xlabel('t')
plt.ylabel('Pression')

""" 

fig3 = plt.figure()
plt.plot(massConcerv.tList, rho[:, -1])
plt.xlabel('t')
plt.ylabel('rho')

fig4 = plt.figure()
plt.plot(massConcerv.x, rho[-1])
plt.xlabel('x')
plt.ylabel('rho') """


plt.show()

print(rho)