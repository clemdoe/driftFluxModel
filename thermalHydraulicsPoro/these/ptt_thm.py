#PTT THM
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

height = 2

P_num = [10833885.0, 10830120.0, 10826355.0, 10822590.0, 10818825.0, 10815060.0, 10811295.0, 10807530.0, 10803765.0, 10800000.0]
P_num = [10845499.0, 10840701.0, 10835839.0, 10830913.0, 10825923.0, 10820868.0, 10815748.0, 10810564.0, 10805314.0, 10800000.0]
P_num=  [10844935.0, 10840209.0, 10835417.0, 10830557.0, 10825631.0, 10820638.0, 10815579.0, 10810452.0, 10805260.0, 10800000.0]

Z = np.linspace((height/10),height, 10)
print(Z)

a_ = - 865
b_ = - 24000
c_ = 10851460#10800000 - a_*2^2 - b_*2
print(f'c: {c_}')
def P(z):
    return a_*z**2 + b_*z + c_
P_anal = [ P(z) for z in Z]
equation_anal = f"y = {a_} * x^2 + {b_} * x + {c_}"


# Interpolation quadratique
quadratic_interp = interp1d(Z, P_num, kind='quadratic')
# Ajustement quadratique (régression quadratique)
coeffs = np.polyfit(Z, P_num, 2)  # Coefficients de l'équation y = ax^2 + bx + c
# Équation obtenue
equation_THM = f"y = {coeffs[0]:.2f} * x^2 + {coeffs[1]:.2f} * x + {coeffs[2]:.2f}"

print(P_anal)
print(P_num)

plt.plot(Z, P_num, label=f'PTT THM: {equation_THM}', marker='o')
plt.plot(Z, P_anal, label=f'P analytique: {equation_anal}', marker='x')
plt.xlabel('Hauteur (m)')
plt.ylabel('Pression (Pa)')
plt.legend()
plt.show()

Z = np.linspace(0 , height -(height/10), 10)
Z_anal = np.linspace(0 ,height, 100)
V_num = [8.34608936, 8.49826145, 8.65043449, 8.80260658, 8.95477962, 9.10695267, 9.25912476, 9.41129875, 9.56346989, 9.71564293]
D_num = [814.866699, 800.275513, 786.197571, 772.606384, 759.477112, 746.786621, 734.513306, 722.636780, 711.138367, 700.000000]

a = ((1/700-1/830)*(1/2))
b = 1/830

V_anal = [ 6927.26439 * (a * z + b) for z in Z_anal]


plt.plot(Z, V_num, label=f'V THM', marker='o')
plt.plot(Z_anal, V_anal, label=f'V analytique', marker='x')
plt.xlabel('Hauteur (m)')
plt.ylabel('Vitesse (M/s)')
plt.legend()
plt.show()


Z = np.linspace(0 + (height/10), height , 10)
Z_anal = np.linspace(0,height, 100)
D_anal = [ 1/(a * z + b) for z in Z_anal]
#Z_anal = np.linspace(0  + +(height/10) /2 ,height - +(height/10)/2, 100)
plt.plot(Z, D_num, label=f'D THM', marker='o')
plt.plot(Z_anal, D_anal, label=f'D analytique', marker='x')
plt.xlabel('Hauteur (m)')
plt.ylabel('Densité (kg/m³)')
plt.legend()
plt.show()

DCOOL = [814.866699, 800.275513, 786.197571, 772.606384, 759.477112, 746.786621, 734.513306, 722.636780, 711.138367, 700.000000]
VCOOL = [8.34608936, 8.49826145, 8.65043449, 8.80260658, 8.95477962, 9.10695267, 9.25912476, 9.41129875, 9.56346989, 9.71564293]

Q = [VCOOL[i]*DCOOL[i] for i in range(len(DCOOL))]
plt.plot(Z, Q, label=f'Q THM', marker='o')
plt.xlabel('Hauteur (m)')
plt.ylabel('Débit (kg/m²/s)')
plt.legend()
plt.show()

print(f'Qmoyen: {np.mean(Q)}')


""" # Données
y = np.linspace(0, 2, 10)  # Ordonnée de 0 à 2 m (données échantillonnées)
PCOOL = np.array([10844935.0, 10840209.0, 10835417.0, 10830557.0, 10825631.0, 
                  10820638.0, 10815579.0, 10810452.0, 10805260.0, 10800000.0])
DCOOL = np.array([814.866699, 800.275513, 786.197571, 772.606384, 759.477112, 
                  746.786621, 734.513306, 722.636780, 711.138367, 700.000000])
VCOOL = np.array([8.34608936, 8.49826145, 8.65043449, 8.80260658, 8.95477962, 
                  9.10695267, 9.25912476, 9.41129875, 9.56346989, 9.71564293])
TCOOL = np.array([556.091797, 558.621033, 561.124817, 563.603149, 566.055176, 
                  568.479553, 570.875427, 573.241882, 575.577515, 577.881409])

# Premier graphe : Pression et Densité
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel("Ordonnée (m)")
ax1.set_ylabel("Pression (Pa)", color="tab:blue")
ax1.plot(y, PCOOL, "o-", color="tab:blue", label="Pression", marker="o")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Densité (kg/m³)", color="tab:red")
ax2.plot(y, DCOOL, "s-", color="tab:red", label="Densité", marker="+")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.grid()
plt.show()

# Deuxième graphe : Vitesse et Température
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel("Ordonnée (m)")
ax1.set_ylabel("Vitesse (m/s)", color="tab:green")
ax1.plot(y, VCOOL, "o-", color="tab:green", label="Vitesse", marker="o")
ax1.tick_params(axis="y", labelcolor="tab:green")

ax2 = ax1.twinx()
ax2.set_ylabel("Température (K)", color="tab:purple")
ax2.plot(y, TCOOL, "s-", color="tab:purple", label="Température", marker="+")
ax2.tick_params(axis="y", labelcolor="tab:purple")

plt.grid()
plt.show() """




