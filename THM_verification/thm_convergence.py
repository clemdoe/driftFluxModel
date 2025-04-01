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

import numpy as np
import matplotlib.pyplot as plt

# Données issues du tableau
categories = ['Pressure', 'Void fraction', 'Temperature']

# ΔRMS
rms_10 = [3.07, 0.024, 0.341]   # 10 kW
rms_35 = [9.55, 0.101, 1.63]    # 35 kW

# Δmax
max_10 = [4.00, 0.063, 1.13]
max_35 = [12.5, 0.264, 3.15]

# Δavg
avg_10 = [2.85, 0.009, 0.225]
avg_35 = [8.77, 0.088, 1.52]

x = np.arange(len(categories))  # positions sur l'axe x
width = 0.35                    # largeur des barres

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# --- 1) Sous-graphe pour ΔRMS ---
axes[0].bar(x - width/2, rms_10, width, label='10 kW', color='#B0CDD9')
axes[0].bar(x + width/2, rms_35, width, label='35 kW', color='#f25E5E')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, rotation=15)
axes[0].set_title('ΔRMS')
axes[0].legend()
axes[0].set_ylabel('Valeur')

# --- 2) Sous-graphe pour Δmax ---
axes[1].bar(x - width/2, max_10, width, label='10 kW', color='#B0CDD9')
axes[1].bar(x + width/2, max_35, width, label='35 kW', color='#f25E5E')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, rotation=15)
axes[1].set_title('Δmax')
axes[1].legend()

# --- 3) Sous-graphe pour Δavg ---
axes[2].bar(x - width/2, avg_10, width, label='10 kW', color='#B0CDD9')
axes[2].bar(x + width/2, avg_35, width, label='35 kW', color='#f25E5E')
axes[2].set_xticks(x)
axes[2].set_xticklabels(categories, rotation=15)
axes[2].set_title('Δavg')
axes[2].legend()

plt.tight_layout()
plt.show()



# Données issues du tableau
categories = ['ΔRMS', 'ΔMAX', 'ΔAVG']

####
# Pressure
P_10 = [3.07, 4.00, 2.85]   # 10 kW
P_35 = [9.55, 12.5, 8.77]    # 35 kW

# Void Fraction
VF_10 = [0.024, 0.063, 0.009]
VF_35 = [0.101, 0.264, 0.088]

# Temperature
T_10 = [0.341, 1.13, 0.225]
T_35 = [1.63, 3.15, 1.52]

x = np.arange(len(categories))  # positions sur l'axe x
width = 0.35                    # largeur des barres

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# --- 1) Sous-graphe pour ΔRMS ---
axes[0].bar(x - width/2, P_10, width, label='10 kW', color='#B0CDD9')
axes[0].bar(x + width/2, P_35, width, label='35 kW', color='#f25E5E')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, rotation=15)
axes[0].set_title('Pressure')
axes[0].legend()
axes[0].set_ylabel('Ecart %')

# --- 2) Sous-graphe pour Δmax ---
axes[1].bar(x - width/2, VF_10, width, label='10 kW', color='#B0CDD9')
axes[1].bar(x + width/2, VF_35, width, label='35 kW', color='#f25E5E')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, rotation=15)
axes[1].set_title('Void Fraction')
axes[1].legend()
axes[1].set_ylabel('Ecart %')

# --- 3) Sous-graphe pour Δavg ---
axes[2].bar(x - width/2, T_10, width, label='10 kW', color='#B0CDD9')
axes[2].bar(x + width/2, T_35, width, label='35 kW', color='#f25E5E')
axes[2].set_xticks(x)
axes[2].set_xticklabels(categories, rotation=15)
axes[2].set_title('Temperature')
axes[2].legend()
axes[2].set_ylabel('Ecart %')

plt.tight_layout()
plt.show()
