# Authors : Clément Huet

from class_MVF import FVM
import numpy as np
from iapws import IAPWS97
import csv
import matplotlib.pyplot as plt
from function import *
from fonctionSousRelax import *

print("Begin of the program")

nTest = 8
powerImposed = 800000000

listSousRelaxFactor = [0.02, 0.04, 0.06, 0.1, 0.12, 0.16, 0.3, 0.5]
testResult = []

for i in range(nTest):
    print(f"sousRelaxationFactor: {listSousRelaxFactor[i]}")
    U, P, H, epsilon, rho_l, rho_g, rho, x_th, zList, rho_g_RES, rho_l_RES, espilon_RES, x_th_RES, I, Hlsat, Hgsat, T, Tsat =  functionRecuperation(listSousRelaxFactor[i], powerImposed)
    testResult.append([U, P, H, epsilon, rho_l, rho_g, rho, x_th, zList, rho_g_RES, rho_l_RES, espilon_RES, x_th_RES, I, Hlsat, Hgsat, T, Tsat])

print(testResult)

fig1 = plt.figure()
for i in range(nTest):
    plt.plot(testResult[i][13], testResult[i][9], label=f"r = {listSousRelaxFactor[i]}")
plt.xlabel("Iteration")
plt.ylabel("rho_g residual")
plt.title("Residus de la densité de la vapeur en fonction du nombre d'itération")
plt.legend()

fig2 = plt.figure()
for i in range(nTest):
    plt.plot(testResult[i][13], testResult[i][10], label=f"r = {listSousRelaxFactor[i]}")
plt.xlabel("Iteration")
plt.ylabel("rho_l residual")
plt.title("Residus de la densité du liquide en fonction du nombre d'itération")
plt.legend()

fig3 = plt.figure()
for i in range(nTest):
    plt.plot(testResult[i][13], testResult[i][11], label=f"r = {listSousRelaxFactor[i]}")
plt.xlabel("Iteration")
plt.ylabel("epsilon residual")
plt.title("Residus de la fraction de vide en fonction du nombre d'itération")
plt.legend()

fig4 = plt.figure()
for i in range(nTest):
    plt.plot(testResult[i][13], testResult[i][12], label=f"r = {listSousRelaxFactor[i]}")
plt.xlabel("Iteration")
plt.ylabel("x_th residual")
plt.title("Residus de la qualité en fonction du nombre d'itération")
plt.legend()

fig5 =plt.figure()
for i in range(nTest):
    if 200 not in  testResult[i][13]:
        plt.plot(testResult[i][8], testResult[i][0], label=f"r = {listSousRelaxFactor[i]}")
plt.xlabel("z (m)")
plt.ylabel("U (m/s)")
plt.title("Vitesse en fonction de la hauteur pour un chauffage de 0.8GW")
plt.legend()

fig6 = plt.figure()
for i in range(nTest):
    if 200 not in testResult[i][13]:
        plt.plot(testResult[i][8], testResult[i][1], label=f"r = {listSousRelaxFactor[i]}")
plt.xlabel("z (m)")
plt.ylabel("P (MPa)")
plt.title("Pression en fonction de la hauteur pour un chauffage de 0.8GW")
plt.legend()

fig7 = plt.figure()
for i in range(nTest):
    if 200 not in testResult[i][13]:
        plt.plot(testResult[i][8], testResult[i][2], label=f"r = {listSousRelaxFactor[i]}")
plt.plot(testResult[1][8], testResult[1][14], label="Hlsat")
plt.plot(testResult[1][8], testResult[1][15], label="Hgsat")
plt.xlabel("z (m)")
plt.ylabel("H (J/kg)")
plt.title("Enthalpie en fonction de la hauteur pour un chauffage de 0.8GW")
plt.legend()

fig8 = plt.figure()
for i in range(nTest):
    if 200 not in testResult[i][13]:
        plt.plot(testResult[i][8], testResult[i][3], label=f"r = {listSousRelaxFactor[i]}")
plt.xlabel("z (m)")
plt.ylabel("epsilon")
plt.title("Fraction de vide en fonction de la hauteur pour un chauffage de 0.8GW")
plt.legend()

fig9 = plt.figure()
for i in range(nTest):
    if 200 not in testResult[i][13]:
        plt.plot(testResult[i][8], testResult[i][16], label=f"r = {listSousRelaxFactor[i]}")
plt.plot(testResult[1][8], testResult[1][17], label="Tsat")
plt.xlabel("z (m)")
plt.ylabel("T (K)")
plt.title("Température en fonction de la hauteur pour un chauffage de 0.8GW")
plt.legend()

plt.show()

