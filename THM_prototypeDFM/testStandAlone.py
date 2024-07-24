from  classDFM import DFMclass
from FVM import FVM
from iapws import IAPWS97
import matplotlib.pyplot as plt
import  numpy as np
from THM_MONO import FVM_ConvectionInCanal_MONO as FVM_Canal_MONO

##Parameters
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

Poro = 0.5655077285
cote = 0.0157 * np.sqrt(Poro)

# Create an object of the class DFMclass
print(f'Tinlet : {T_inlet} K')
print(f'Pinlet : {P_inlet} Pa')
print(f'hinlet : {h_inlet} J/kg')
print(f'uinlet : {u_inlet} m/s')
print(f'Poutlet : {P_outlet} Pa')
print(f'height : {height} m')
print(f'fuelRadius : {fuelRadius} m')
print(f'cladRadius : {cladRadius} m')
print(f'cote : {cote} m')
dfm = DFMclass(nCells, T_inlet, P_inlet, u_inlet, P_outlet, height, fuelRadius, cladRadius, cote, 'FVM', 'base', 'base', 'GEramp')
              #  nCells, tInlet, pInlet, uInlet, pOutlet, height, fuelRadius, cladRadius, waterGap,  numericalMethod, frfaccorel, P2P2corel, voidFractionCorrel):
        
dfm.set_Fission_Power(300000000, 'constant', 0, height)
# Resolve the DFM
dfm.resolveDFM()
print(f'Pressure: {dfm.P[-1]} Pa')
print(f'Enthalpy: {dfm.H[-1]} J/kg')
Tsurf = dfm.compute_T_surf()
print(f'Temperature at the surface: {Tsurf} K')
print(f'Temperature of water: {dfm.T_water} K')

Qflow = u_inlet * 1000

convection_sol = FVM_Canal_MONO(height, T_inlet, Qflow, P_inlet/1000000, nCells, "cylindrical", fuelRadius, cladRadius, cote )
convection_sol.set_Fission_Power(300000000, 'constant')
print(f'convection.qfluid: {convection_sol.q_fluid[1]*convection_sol.dz/(convection_sol.Q_flow*convection_sol.A_canal)}')
for i in range(1,convection_sol.N_vol-1):
                
    convection_sol.set_ADi_conv(i, ci=-1, ai=1, bi=0, di = convection_sol.q_fluid[i]*convection_sol.dz/(convection_sol.Q_flow*convection_sol.A_canal))

    A0,Am1 = np.zeros(convection_sol.N_vol), np.zeros(convection_sol.N_vol)
    A0[0] = 1
    D0 = convection_sol.h_z0 + convection_sol.q_fluid[0]*convection_sol.dz/(2*convection_sol.Q_flow*convection_sol.A_canal)
    Am1[-2:]=[-1, 1]
    Dm1 = convection_sol.q_fluid[-1]*convection_sol.dz/(convection_sol.Q_flow*convection_sol.A_canal)
    convection_sol.set_CL_conv(A0,Am1,D0,Dm1)

print("$---------- Solving for h(z) using the Finite Volumes Method.")
convection_sol.h_z = convection_sol.solve_h_in_canal() # solving the enthalpy evolution in the canal
print("$---------- Solving for T_surf(z) using the Dittus-Boelter correlation. Water Properties evaluated by IAPWS97")
Tsurf = convection_sol.compute_T_surf() # computing and retrieving the clad surface temperatures obtained through the Dittus-Boelter correlation
print(f'Temperature at the surface: {Tsurf} K')
print(f'Temperature of water: {convection_sol.T_water} K')