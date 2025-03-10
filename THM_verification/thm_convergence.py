#PTT THM
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re

def extract_pcool_vcool(filename):
    """
    Extrait les valeurs associées à 'PCOOL FINAL' et 'VCOOL FINAL'
    depuis le fichier dont le nom est passé en paramètre.
    
    Retourne un tuple (pcool_final, vcool_final) qui sont des listes de nombres.
    """
    pcool_final = []
    vcool_final = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Recherche de la chaîne 'PCOOL FINAL'
            if "PCOOL FINAL" in line:
                # Si une séparation par deux-points est présente, on utilise le texte après :
                if ":" in line:
                    _, data = line.split(":", 1)
                else:
                    data = line.split("PCOOL FINAL", 1)[-1]
                # Extraction de tous les nombres (y compris en notation scientifique)
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", data)
                pcool_final.extend([float(n) for n in numbers])
                
            # Recherche de la chaîne 'VCOOL FINAL'
            elif "VCOOL FINAL" in line:
                if ":" in line:
                    _, data = line.split(":", 1)
                else:
                    data = line.split("VCOOL FINAL", 1)[-1]
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", data)
                vcool_final.extend([float(n) for n in numbers])
    
    return pcool_final, vcool_final

def erreur_absolue_moyenne(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    return np.mean(np.abs(l1 - l2))

height = 2
a_ = - 865
b_ = - 24000
c_ = 10851460#10800000 - a_*2^2 - b_*2
print(f'c: {c_}')
def P(z):
    return a_*z**2 + b_*z + c_

a = ((1/700-1/830)*(1/2))
b = 1/830
def V(z):
    return 6927.26439 * (a * z + b)


# Exemple d'utilisation:
volumes = [4,6,8,10,12,18, 30, 100]
pcool = []
vcool = []
for i in range(len(volumes)):
    p, v = extract_pcool_vcool(f"BWR\driftFluxModel\THM_verification\THM_{volumes[i]}_200.result")
    pcool.append(p)
    vcool.append(v)
    print("PCOOL FINAL :", pcool)
    print("VCOOL FINAL :", vcool)

error_list_V = []
error_list_P = []
for i in range(len(volumes)):
    Z_p = np.linspace((height/volumes[i]),height, volumes[i])
    Z_v = np.linspace(0 , height -(height/volumes[i]), volumes[i])
    P_num = pcool[i]
    V_num = vcool[i]
    P_anal = [ P(z) for z in Z_p]
    V_anal = [ V(z) for z in Z_v]
    error_list_P.append(erreur_absolue_moyenne(P_num, P_anal))
    error_list_V.append(erreur_absolue_moyenne(V_num, V_anal))

print(f'Error liste: {error_list_P}')
print(f'Error liste: {error_list_V}')
fig1, ax1 = plt.subplots()
ax1.plot(volumes, error_list_P, label=f'Erreur absolue moyenne sur la pression (Pa)')
ax1.set_xlabel('Nombre de volumes')
ax1.set_ylabel('Erreur absolue moyenne')
plt.legend()
plt.grid()

fig2, ax2 = plt.subplots()
ax2.plot(volumes, error_list_V, label=f'Erreur absolue moyenne sur la vitesse (m/s)')
ax2.set_xlabel('Nombre de volumes')
ax2.set_ylabel('Erreur absolue moyenne')
plt.legend()
plt.grid()
plt.show()

log_error_list_P = np.log(error_list_P)
log_error_list_V = np.log(error_list_V)
log_Z = np.log(volumes)

coeffs_P = np.polyfit(log_Z, log_error_list_P, 1)
coeffs_V = np.polyfit(log_Z, log_error_list_V, 1)

fig3, ax3 = plt.subplots()
ax3.plot(log_Z, log_error_list_P, label=f'Erreur absolue moyenne sur la pression. \n Ordre de convergence: {coeffs_P[0]:.2f}')
ax3.set_xlabel('Log(Nombre de volumes)')
ax3.set_ylabel('Log(Erreur absolue moyenne)')
plt.legend()
plt.grid()

fig4, ax4 = plt.subplots()
ax4.plot(log_Z, log_error_list_V, label=f'Erreur absolue moyenne sur la vitesse. \n Ordre de convergence: {coeffs_V[0]:.2f}')
ax4.set_xlabel('Log(Nombre de volumes)')
ax4.set_ylabel('Log(Erreur absolue moyenne)')
plt.legend()
plt.grid()
plt.show()

print(f'pcool 100:{pcool[-1]}')
print(f'vcool 100:{vcool[-1]}')
