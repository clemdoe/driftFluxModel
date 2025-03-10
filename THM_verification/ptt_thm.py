#PTT THM
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

height = 2

P_num = [10833885.0, 10830120.0, 10826355.0, 10822590.0, 10818825.0, 10815060.0, 10811295.0, 10807530.0, 10803765.0, 10800000.0]
P_num = [10845499.0, 10840701.0, 10835839.0, 10830913.0, 10825923.0, 10820868.0, 10815748.0, 10810564.0, 10805314.0, 10800000.0]
P_num=  [10844935.0, 10840209.0, 10835417.0, 10830557.0, 10825631.0, 10820638.0, 10815579.0, 10810452.0, 10805260.0, 10800000.0]
V_100 = [8.34608936, 8.36155987, 8.37703133, 8.39250374, 8.40797424, 8.4234457, 8.43891621, 8.45438671, 8.46985817, 8.48533058, 8.50080109, 8.51627254, 8.53174305, 8.54721451, 8.56268501, 8.57815742, 8.59362793, 8.60909939, 8.62456989, 8.6400404, 8.65551281, 8.67098331, 8.68645573, 8.70192623, 8.71739674, 8.73286819, 8.74833965, 8.76381016, 8.77928162, 8.79475212, 8.81022358, 8.82569504, 8.8411665, 8.856637, 8.87210846, 8.88757992, 8.90305042, 8.91852188, 8.93399239, 8.94946384, 8.96493626, 8.98040676, 8.99587727, 9.01134872, 9.02681923, 9.04229069, 9.05776215, 9.0732336, 9.08870506, 9.10417557, 9.11964607, 9.13511753, 9.15058899, 9.16606045, 9.18153095, 9.19700241, 9.21247387, 9.22794437, 9.24341488, 9.25888729, 9.27435875, 9.28982925, 9.30530071, 9.32077122, 9.33624268, 9.35171413, 9.36718559, 9.3826561, 9.39812756, 9.41359806, 9.42906952, 9.44454098, 9.46001244, 9.47548389, 9.4909544, 9.50642586, 9.52189636, 9.53736782, 9.55283928, 9.56831074, 9.58378124, 9.59925175, 9.61472321, 9.63019466, 9.64566612, 9.66113758, 9.67660809, 9.69207954, 9.70755005, 9.72302055, 9.73849297, 9.75396347, 9.76943493, 9.78490543, 9.80037689, 9.81584835, 9.83131886, 9.84679127, 9.86226177, 9.87773323]
P_100 = [10850785.0, 10850306.0, 10849826.0, 10849345.0, 10848864.0, 10848382.0, 10847899.0, 10847416.0, 10846932.0, 10846447.0, 10845962.0, 10845476.0, 10844989.0, 10844502.0, 10844013.0, 10843525.0, 10843035.0, 10842545.0, 10842054.0, 10841562.0, 10841070.0, 10840577.0, 10840083.0, 10839589.0, 10839094.0, 10838598.0, 10838102.0, 10837605.0, 10837107.0, 10836608.0, 10836109.0, 10835609.0, 10835109.0, 10834607.0, 10834105.0, 10833603.0, 10833099.0, 10832595.0, 10832091.0, 10831585.0, 10831079.0, 10830572.0, 10830065.0, 10829557.0, 10829048.0, 10828538.0, 10828028.0, 10827517.0, 10827006.0, 10826493.0, 10825980.0, 10825467.0, 10824952.0, 10824437.0, 10823922.0, 10823405.0, 10822888.0, 10822370.0, 10821852.0, 10821333.0, 10820813.0, 10820292.0, 10819771.0, 10819249.0, 10818726.0, 10818203.0, 10817679.0, 10817154.0, 10816629.0, 10816103.0, 10815576.0, 10815049.0, 10814520.0, 10813992.0, 10813462.0, 10812932.0, 10812401.0, 10811869.0, 10811337.0, 10810804.0, 10810271.0, 10809736.0, 10809201.0, 10808665.0, 10808129.0, 10807592.0, 10807054.0, 10806516.0, 10805976.0, 10805437.0, 10804896.0, 10804355.0, 10803813.0, 10803270.0, 10802727.0, 10802183.0, 10801638.0, 10801093.0, 10800547.0, 10800000.0]

Z = np.linspace((height/10),height, 10)
Z_100 = np.linspace((height/100) , height, 100)
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

plt.scatter(Z, P_num, label=f'P THM_10: {equation_THM}', marker='o')
plt.scatter(Z_100, P_100, label=f'P THM_100', marker='+', color="green")
plt.plot(Z, P_anal, label=f'P analytique: {equation_anal}', color="red")
plt.xlabel('Hauteur (m)')
plt.ylabel('Pression (Pa)')
plt.legend()
plt.show()

Z = np.linspace(0 , height -(height/10), 10)
Z_anal = np.linspace(0 ,height, 100)
Z_100 = np.linspace(0 , height, 100)
V_num = [8.34608936, 8.49826145, 8.65043449, 8.80260658, 8.95477962, 9.10695267, 9.25912476, 9.41129875, 9.56346989, 9.71564293]
D_num = [814.866699, 800.275513, 786.197571, 772.606384, 759.477112, 746.786621, 734.513306, 722.636780, 711.138367, 700.000000]

a = ((1/700-1/830)*(1/2))
b = 1/830

V_anal = [ 6927.26439 * (a * z + b) for z in Z_anal]


plt.plot(Z, V_num, label=f'V THM', marker='o')
plt.scatter(Z_100, V_100, label=f'V THM_100', marker='+', color="green")
plt.plot(Z_anal, V_anal, label=f'V analytique', color="red")
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




