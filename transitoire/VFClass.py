import numpy as np

class VF_1D:
    def __init__(self, L, N_vol):
        self.L, self.N_vol = L, N_vol
        self.dx = L/self.N_vol
        self.x = np.linspace(0.5*self.dx, L-0.5*self.dx, self.N_vol)
        self.A, self.D = np.eye(self.N_vol), np.zeros(self.N_vol)
        return

    def set_ADi(self, i, ci, ai, bi, di):
        self.A[i, i-1:i+2] = [ci, ai, bi]
        self.D[i] = di
        return
    
    def set_CL(self, A0, Am1, D0, Dm1):
        self.A[0], self.A[-1] = A0, Am1
        self.D[0], self.D[-1] = D0, Dm1
        return
    
    def fillingOutsideBoundary(self, i, j, ai, bi):
        self.A[i, j:j+2] = [ai, bi]
        return 

    def resoudre_X(self):
        return np.linalg.solve(self.A, self.D) 
    
    def set_transitoire(self, t_tot, Xini, dt):
        self.t_tot, self.dt = t_tot, dt           
        self.N_temps = round(self.t_tot / self.dt) # pas de temps (timesteps), il faut etre un nombre entier
        self.tList = np.linspace(0, self.t_tot, self.N_temps) # liste de temps
        self.X = np.zeros((self.N_temps, self.N_vol)) # tableau 2D de temperature. 
        self.X[0] = Xini # Tini est une liste
        return 
        
    