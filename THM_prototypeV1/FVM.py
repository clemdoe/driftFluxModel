import numpy as np


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

