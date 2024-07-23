# This file contains the class FVM and the class plotting to solve a differential equation and plot the results
# Used to calculate the drift velocity of the mixture in the fuel assembly
# Authors : Clément Huet

import numpy as np
from iapws import IAPWS97
import matplotlib.pyplot as plt


#class to solve a differential equation. In this project it can solve differention equation of enthalpy, pressure, velocity
#The parameters are:
#ai, bi, ci, di: the parameters of the differential equation
#A00, A01, Am0, Am1, D0, Dm1: the boundary conditions
#N_vol: the number of volumes
#H: the height of the fuel rod
class FVM:
    def __init__(self, A00, A01, Am0, Am1, D0, Dm1, N_vol, H):
        self.A00 = A00
        self.A01 = A01
        self.Am0 = Am0
        self.Am1 = Am1
        self.D0 = D0
        self.Dm1 = Dm1
        self.N_vol = N_vol
        self.H = H
        self.dz = H / N_vol
        self.z = np.linspace(0, H, self.N_vol)
        self.A, self.D = np.eye(self.N_vol), np.zeros(self.N_vol)

    #function to set the matrix A and D
    def set_ADi(self, i, ci, ai, bi, di):
        self.A[i, i-1:i+2] = [ci, ai, bi]
        self.D[i] = di
        return
    
    #function to set the boundary conditions
    def set_CL(self, A0, Am1, D0, Dm1):
        self.A[0], self.A[-1] = A0, Am1
        self.D[0], self.D[-1] = D0, Dm1
        return
    
    #function to solve the system of equations
    def resoudre_h(self):
        return np.linalg.solve(self.A, self.D) 
    
    #function to set the transient parameters
    """ def set_transitoire(self, t_tot, Tini, dt):
        self.t_tot, self.dt = t_tot, dt           
        self.N_temps = round(self.t_tot / self.dt) # pas de temps (timesteps), il faut etre un nombre entier
        self.T = np.zeros((self.N_temps, self.N_vol)) # tableau 2D de temperature. 
        self.T[0] = Tini # Tini est une liste de temperature initiale """
        
    #function to fill the matrix A and D
    # def AD_filling(self):
    #     for i in range(1, self.N_vol-1):
    #         self.set_ADi(i, ci = self.ci,
    #         ai = self.ai,
    #         bi = self.bi,
    #         di = self.di )
    
    #function to set the boundary conditions
    def boundaryFilling(self):
        A0, Am1 = np.zeros(self.N_vol), np.zeros(self.N_vol)
        A0[:2] = [self.A00, self.A01]
        Am1[-2:] = [self.Am0, self.Am1]
        D0 = self.D0
        Dm1 = self.Dm1
        self.set_CL(A0, Am1, D0, Dm1)

    def fillingOutside(self, i, j, ci, ai, bi):
        self.A[i, j-1:j+2] = [ci, ai, bi]
        return 
    
    def fillingOutsideBoundary(self, i, j, ai, bi):
        self.A[i, j:j+2] = [ai, bi]
        return 
    
    #function to solve the differential equation of enthalppie
    def differential(self):
        self.boundaryFilling()
        self.h = self.resoudre_h()
        return self.h
    
    #function to calculate the temperature of the surface of the fuel using the fluid parameters, the distribution of enthalpie and the heat flux
    def verticalResolution(self):
        self.differential()
    
    
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