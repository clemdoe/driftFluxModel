from  classDFM import DFMclass
from FVM import FVM
from iapws import IAPWS97
import matplotlib.pyplot as plt
import  numpy as np


#Boundary conditions
T_inlet = 602.75 #K
P_inlet = 14.9 * (10**6) #Pa
h_inlet = IAPWS97(T = T_inlet, P = P_inlet * 10**(-6)).h * 1000 #J/kg
u_inlet = 4.68292412 #m/s
P_outlet =  14739394.95 #Pa
nCells = 30

#Geometry parameters
height = 1.655 #m
fuelRadius = 5.6*10**(-3) #External radius of the fuel m
cladRadius = 6.52*10**(-3) #External radius of the clad m
waterGap = 0.5*10**(-3) #Gap between the clad and the water m

DFM1 = DFMclass(nCells, u_inlet, P_outlet, h_inlet, height, fuelRadius, cladRadius, waterGap, 'FVM', 'base', 'base', 'GEramp', 900000000, 'sinusoidal')
DFM2 = DFMclass(nCells, u_inlet, P_outlet, h_inlet, height, fuelRadius, cladRadius, waterGap, 'FVM', 'base', 'base', 'GEramp', 1000000000, 'sinusoidal')
DFM3 = DFMclass(nCells, u_inlet, P_outlet, h_inlet, height, fuelRadius, cladRadius, waterGap, 'FVM', 'base', 'base', 'GEramp', 1500000000, 'sinusoidal')
#DFM2 = DFMclass(nCells, u_inlet, P_outlet, h_inlet, height, fuelRadius, cladRadius, waterGap, 'FVM', 'base', 'base', 'modBestion')
DFM1.resolveDFM()
DFM2.resolveDFM()
DFM3.resolveDFM()
#DFM2.resolveDFM()
DFM1.createBoundaryEnthalpy()
h = DFM1.H[-1]/1000
h2 = DFM2.H[-1]/1000
h3 = DFM3.H[-1]/1000
eps = DFM1.voidFraction[-1]
eps2 = DFM2.voidFraction[-1]
eps3 = DFM3.voidFraction[-1]
#eps2 = DFM2.voidFraction[-1]
z = DFM1.zList

T1 = []
Tsat1 = []
T2 = []
Tsat2 = []
T3 = []
Tsat3 = []
for i in range(nCells):
    T1.append(IAPWS97(P = DFM1.P[-1][i] * 10**(-6), h = h[i], x = 0).T - 273.5)
    Tsat1.append(IAPWS97(P = DFM1.P[-1][i] * 10**(-6), x = 0).T - 273.5)
    T2.append(IAPWS97(P = DFM2.P[-1][i] * 10**(-6), h = h2[i], x = 0).T - 273.5)
    Tsat2.append(IAPWS97(P = DFM2.P[-1][i] * 10**(-6), x = 0).T - 273.5)
    T3.append(IAPWS97(P = DFM3.P[-1][i] * 10**(-6), h = h3[i], x = 0).T - 273.5)
    Tsat3.append(IAPWS97(P = DFM3.P[-1][i] * 10**(-6), x = 0).T - 273.5)

#print(f'U: {DFM1.U}, P: {DFM1.P}, H: {DFM1.H}, epsilon: {DFM1.voidFraction} rho: {DFM1.rho}, rhoG: {DFM1.rhoG}, rhoL: {DFM1.rhoL}, xTh: {DFM1.xTh}, f: {DFM1.f}, Vgj: {DFM1.Vgj}, Vgj_prime: {DFM1.VgjPrime}, C0: {DFM1.C0}')
print(f'T: {T1} \n U: {list(DFM1.U[-1])}, \n P: {list(DFM1.P[-1])}, \n H: {list(DFM1.H[-1])}, \n epsilon: {list(DFM1.voidFraction[-1])}, \n rho: {list(DFM1.rho[-1])}, \n rhoG: {list(DFM1.rhoG[-1])}, \n rhoL: {list(DFM1.rhoL[-1])}, \n xTh: {list(DFM1.xTh[-1])}, \n f: {list(DFM1.f[-1])}, \n Vgj: {list(DFM1.Vgj[-1])}, \n Vgj_prime: {list(DFM1.VgjPrime[-1])}, \n C0: {list(DFM1.C0[-1])}')

print(f'z: {list(z)}') 

fig1 = plt.figure()
plt.plot(z, h, label="Q = 500 MW")
plt.plot(z, h2, label="Q = 1000 MW")
plt.plot(z, h3, label="Q = 1500 MW")
plt.plot(z,  DFM1.hlSat, label="hlSat")

plt.xlabel('Height (m)')
plt.ylabel('Enthalpy (k J/kg)')
plt.title('Enthalpy distribution')
plt.legend()

fig2 = plt.figure()
plt.plot(DFM1.I, DFM1.rhoGResiduals, label="Residuals rhoG")
plt.plot(DFM1.I, DFM1.rhoLResiduals, label="Residuals rhoL")
plt.xlabel('Iteration')
plt.ylabel('Residuals')
plt.legend()

fig3 = plt.figure()
plt.plot(DFM1.I, DFM1.rhoResiduals)
plt.xlabel('Iteration')
plt.ylabel('rho Residuals')
plt.legend()

fig4 = plt.figure()
plt.plot(DFM1.I, DFM1.EPSresiduals)
plt.xlabel('Iteration')
plt.ylabel('Epsilon Residuals')
plt.legend()

fig5 = plt.figure()
plt.plot(DFM1.I, DFM1.xThResiduals)
plt.xlabel('Iteration')
plt.ylabel('xTh Residuals')
plt.legend()

fig6 = plt.figure()
plt.plot(z, eps, label="Q = 500 MW")
plt.plot(z, eps2, label="Q = 1000 MW")
plt.plot(z, eps3, label="Q = 1500 MW")
#plt.plot(z, eps2, label="GEramp")
plt.xlabel('Height (m)')
plt.ylabel('Void fraction')
plt.title('Void fraction distribution')
plt.legend()

fig7 = plt.figure()
plt.plot(z, DFM1.rho[-1], label="Q = 500 MW")
plt.plot(z, DFM2.rho[-1], label="Q = 1000 MW")
plt.plot(z, DFM3.rho[-1], label="Q = 1500 MW")
plt.xlabel('Height (m)')
plt.ylabel('Density (kg/m3)')
plt.title('Density distribution')
plt.legend()

fig8 = plt.figure()
plt.plot(z, DFM1.rhoG[-1])
plt.xlabel('Height (m)')
plt.ylabel('Density gas (kg/m3)')
plt.title('Density gas distribution')
plt.legend()

fig9 = plt.figure()
plt.plot(z, DFM1.rhoL[-1])
plt.xlabel('Height (m)')
plt.ylabel('Density liquid (kg/m3)')
plt.title('Density liquid distribution')
plt.legend()

fig10 = plt.figure()
plt.plot(z, DFM1.U[-1], label="Q = 500 MW")
plt.plot(z, DFM2.U[-1], label="Q = 1000 MW")
plt.plot(z, DFM3.U[-1], label="Q = 1500 MW")
plt.xlabel('Height (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity distribution')
plt.legend()

fig11 = plt.figure()
plt.plot(z, DFM1.P[-1])
plt.xlabel('Height (m)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure distribution')
plt.legend()

fig12 = plt.figure()
plt.plot(z, DFM1.Q, label="Q")
plt.plot(z, DFM1.q__, label="q__")
plt.xlabel('Height (m)')
plt.ylabel('Heat (W/m3)')
plt.title('Heat distribution')

fig13 = plt.figure()
plt.plot(z, T1, label="Q = 500 MW")
plt.plot(z, Tsat1, label="Tsat1")
plt.plot(z, T2, label="Q = 1000 MW")
plt.plot(z, Tsat2, label="Tsat2")
plt.plot(z, T3, label="Q = 1500 MW")
plt.plot(z, Tsat3, label="Tsat3")
plt.xlabel('Height (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature distribution')
plt.legend()

plt.show()

