import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt
from Th_properties import *

class plotter():
    def __init__(self, nCells, zList, U, P, T, rho, rho_g, rho_l, x_th, H, epsilon, epsilonResidual, rhoResidual, xThResidual, I):
        self.hl = []
        self.hg = []
        self.Psat = []
        self.Tsat = []
        self.nCells = nCells
        self.zList = zList
        self.U = U
        self.P = P
        self.T = T
        self.rho = rho
        self.rho_g = rho_g
        self.rho_l = rho_l
        self.x_th = x_th
        self.H = H
        self.epsilon = epsilon
        self.epsilonResidual = epsilonResidual
        self.rhoResidual = rhoResidual
        self.xThResidual = xThResidual
        self.I = I

        for i in range(nCells):
            hltemp = IAPWS97(P = P[i] * 10**(-6), x=0).h
            hgtemp = IAPWS97(P = P[i] * 10**(-6), x=1).h
            self.hl.append(hltemp)
            self.hg.append(hgtemp)

            self.Tsat.append(IAPWS97(P = P[i] * 10**(-6), x = 0.5).T)
            self.Psat.append(IAPWS97(T = T[i], x = 0.5).P)

            H[i] = H[i] / 1000

    def plotFields(self, print_U = False, print_P = False, print_T = False, print_rho = False, print_rho_g = False, print_rho_l = False, print_xth = False, print_H = False, print_epsilon = False, print_epsilon_Res = False, print_rho_Res = False, print_xth_Res = False):

        if print_U:
            fig1 = plt.figure()
            plt.plot(self.zList, self.U, label="U")
            plt.xlabel("z [m]")
            plt.ylabel("U [m/s]")
            plt.legend()

        if print_P:
            fig2 = plt.figure()
            plt.plot(self.zList, self.P, label="P")
            plt.xlabel("z [m]")
            plt.ylabel("P [Pa]")
            plt.legend()
        
        if print_T:
            fig7 = plt.figure()
            plt.plot(self.zList, self.T, label="T_DFM")
            plt.plot(self.zList, self.Tsat, label="Tsat")
            plt.xlabel("z [m]")
            plt.ylabel("T [K]")
            plt.legend()

        if print_rho:
            fig3 = plt.figure()
            plt.plot(self.zList, self.rho, label="rho")
            plt.xlabel("z [m]")
            plt.ylabel("rho [kg/m3]")
            plt.legend()

        if print_rho_g:
            fig4 = plt.figure()
            plt.plot(self.zList, self.rho_g, label="rho_g")
            plt.xlabel("z [m]")
            plt.ylabel("rho_g [kg/m3]")
            plt.legend()

        if print_rho_l:
            fig5 = plt.figure()
            plt.plot(self.zList, self.rho_l, label="rho_l")
            plt.xlabel("z [m]")
            plt.ylabel("rho_l [kg/m3]")
            plt.legend()
        
        if print_xth:
            fig6 = plt.figure()
            plt.plot(self.zList, self.x_th, label="quality")
            plt.xlabel("z [m]")
            plt.ylabel("quality")
            plt.legend()

        if print_H:
            fig12 = plt.figure()
            plt.plot(self.zList, self.H, label="H")
            plt.plot(self.zList, self.hl, label="hl")
            plt.plot(self.zList, self.hg, label="hg")
            plt.xlabel("z [m]")
            plt.ylabel("H [J/kg]")
            plt.legend()
        
        if print_epsilon:
            fig13 = plt.figure()
            plt.plot(self.zList, self.epsilon, label="epsilon")
            plt.xlabel("z [m]")
            plt.ylabel("epsilon")
            plt.legend()

        if print_epsilon_Res:
            fig14 = plt.figure()
            plt.plot(self.I, self.epsilonResidual, label="epsilon")
            plt.xlabel("iteration")
            plt.ylabel("epsilon residual")
            plt.legend()

        if print_rho_Res:
            fig15 = plt.figure()
            plt.plot(self.I, self.rhoResidual, label="rho")
            plt.xlabel("iteration")
            plt.ylabel("rho residual")
            plt.legend()

        if print_xth_Res:
            fig16 = plt.figure()
            plt.plot(self.I, self.xThResidual, label="x_th")
            plt.xlabel("iteration")
            plt.ylabel("x_th residual")
            plt.legend()

        plt.show()

    def printFields(self, print_U = False, print_P = False, print_T = False, print_rho = False, print_rho_g = False, print_rho_l = False, print_xth = False, print_H = False, print_epsilon = False, print_epsilon_Res = False, print_rho_Res = False, print_xth_Res = False):

        print("Printing final fields:")
        if print_U:
            print(f'U = {list(self.U)}')

        if print_P:
            print(f'P = {list(self.P)}')

        if print_T:
            print(f'T = {list(self.T)}')
        
        if print_rho:
            print(f'rho = {list(self.rho)}')

        if print_rho_g:
            print(f'rho_g = {list(self.rho_g)}')

        if print_rho_l:
            print(f'rho_l = {list(self.rho_l)}')

        if print_xth:
            print(f'x_th_new = {list(self.x_th)}')

        if print_H:
            print(f'H = {list(self.H)}')

        if print_epsilon:
            print(f'epsilon = {list(self.epsilon)}')

        if print_epsilon_Res:
            print(f'epsilonResidual = {list(self.epsilonResidual)}')

        if print_rho_Res:
            print(f'rhoResidual = {list(self.rhoResidual)}')

        if print_xth_Res:
            print(f'xThResidual = {list(self.xThResidual)}')

#class to plot the temperature distribution in the fuel rod in the radial or longitudinal direction with different possibilities
#The parameters are:
#convection: the object of the class FVMconvection
#conduction: the object of the class MDFconduction
#type: the type of the plot (radial or longitudinal)
#SlicePlace: the height of the slice in the fuel rod
#temperature_list: the list of the temperature distribution in the fuel rod and in the water canal / surface 
class plotting():
    def __init__(self, convection, conduction, type, SlicePlace, temperature_list):
        self.convection = convection
        self.conduction = conduction
        self.type = type
        self.SlicePlace = SlicePlace
        self.temperature_list = temperature_list
        self.RawRadius = conduction.r_to_plot
        self.rc = conduction.clad_radius
        self.rf = conduction.fuel_radius
        self.rg = conduction.gap_radius
        self.e_canal = conduction.e_canal
        self.rw = conduction.water_radius

    #function to get the index of the value in the list
    def indexation(self, value, L):
        L=list(L)
        if value in L:
            return L.index(value), L.index(value)
        else:
            for i in range(len(L)):
                if L[i]>value:
                    return i-1, i
            
    #function to calculate the temperature at a given height z for every radius
    def calculSlice(self, z, TEMP, coord):
        a = self.indexation(z, coord)
        T1=TEMP[a[0]]
        T2=TEMP[a[1]]
        return (T1 + T2)/2
    
    #function to create the radius array at a given height z
    def createRadius(self):
        self.Radius = list(self.RawRadius)
        self.T=self.calculSlice(self.SlicePlace, self.temperature_list, self.convection.x)
        self.Radius.append(self.rc)
        self.Radius.append(self.rc + self.e_canal / 2)

        Radius = np.array(self.Radius)
        for i in range(len(self.Radius)):
            self.Radius[i] = 1000*self.Radius[i]
        self.Radius = np.array(self.Radius)
        return self.T, self.Radius
    
    #function to plot the temperature distribution in the fuel rod
    def plot(self):
        if self.type == "radial":
            fig , ax = plt.subplots(dpi = 150)
            plt.fill_between(self.Radius, self.T, 0,
                where = (self.Radius >= 0) & (self.Radius <= self.rf*1000),
                color = 'lightcoral', label ="Fuel")
            plt.fill_between(self.Radius, self.T, 0,
                where = (self.Radius >= self.rf*1000) & (self.Radius <= self.rg*1000),
                color = 'bisque', label = "Gap")
            plt.fill_between(self.Radius, self.T, 0,
                where = (self.Radius >= self.rg*1000) & (self.Radius <= self.rc*1000),
                color = 'khaki', label = "Clad")
            plt.fill_between(self.Radius, self.T, 0,
                where = (self.Radius >= self.rc*1000) & (self.Radius <= self.rw*1000),
                color = 'turquoise', label = "Water")
            plt.plot(self.Radius, self.T)
            plt.xlabel('Rayon (mm)')
            plt.ylabel('Temperature (K)')
            plt.title('Temperature en fonction du rayon')
            plt.ylim(400, 1200)
            plt.legend()
            plt.show()
        
        elif self.type == "longitudinal":
            plt.plot(self.convection.x, self.T)
            plt.xlabel('Longueur (m)')
            plt.ylabel('Temperature (K)')
            plt.title('Temperature en fonction de la hauteur')
            plt.show()

    #function to plot the temperature distribution using a scalar field. The temperature is function of the radius and the height
    def plot_scalar_field(self):
        fig , ax = plt.subplots(dpi = 150)
        T, R = self.createRadius()
        Z = self.convection.x
        plt.xlabel('Rayon (mm)')
        plt.ylabel('Hauteur (m)')
        plt.title('Temperature (K) en fonction du rayon et de la hauteur')
        plt.pcolormesh(R, Z, self.temperature_list, cmap = 'plasma') 
        plt.colorbar()
        plt.show()

