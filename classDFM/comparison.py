import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel("BWR\BWR_Project\driftFluxModel\sousRelaxationVSGenFoam\Firstopenfoam.xlsx")

# Create empty lists for each column
columns = df.columns.tolist()
data = [[] for _ in columns]

# Iterate over each row and append values to the corresponding list
for index, row in df.iterrows():
    for i, col in enumerate(columns):
        data[i].append(row[col])

# Plot each column according to the first column
x = data[0]
zList = [0.0, 0.18388888888888888, 0.36777777777777776, 0.5516666666666666, 0.7355555555555555, 0.9194444444444444, 1.1033333333333333, 1.2872222222222223, 1.471111111111111, 1.655]
P = [15100171.403406527, 15100143.543932352, 15068301.916358441, 15036527.240923077, 15004819.418209348, 14969011.810147192, 14924376.833746297, 14870417.536772883, 14808245.753696738, 14739394.95]
U = [4.68292412, 4.682949760205058, 4.678484947475844, 4.674093287340649, 4.669761245200797, 4.947335372522202, 5.4778498353462295, 6.031875447883201, 6.598967627268407, 7.177104699468449]
T = [602.8380050608388, 605.8383544371075, 608.694024496683, 611.4153100988632, 613.9886192097954, 615.1423034622854, 614.9033316519534, 614.6136620897613, 614.2788460466879, 613.9067287213829]
epsilon = [9.996386519850264e-05, 0.000106468129468025, 0.00011297239373754737, 0.00013038050339796988, 0.00015588402263171986, 0.06815459334973951, 0.17730825954141632, 0.27182017040200684, 0.351467592408377, 0.41944922588719125]

for i in range(len(data[7])):
    data[7][i] = (1/0.434492) * data[7][i]

#Pressure
fig1 = plt.figure()
y = data[1]
P_genfoam = y
plt.plot(x, y, label='GeN-FOAM ' + columns[1])
plt.plot(zList, P, label='DFM ' + columns[1])
# Add labels and legend
plt.xlabel(columns[0])
plt.ylabel('Pressure (MPa)')
plt.legend()

print(P_genfoam)

#Temperature
fig2 = plt.figure()
y = data[3]
plt.plot(x, y, label='GeN-FOAM ' + columns[3])
plt.plot(zList, T, label='DFM ' + columns[3])
# Add labels and legend
plt.xlabel(columns[0])
plt.ylabel('Temperature (K)')
plt.legend()

#Velocity
fig3 = plt.figure()
y = data[5]
plt.plot(x, y, label='GeN-FOAM ' + columns[5])
plt.plot(zList, U, label='DFM ' + columns[5])
# Add labels and legend
plt.xlabel(columns[0])
plt.ylabel('Vitesse (m/s)')
plt.legend()    

#Velocity
fig3 = plt.figure()
y = data[7]
plt.plot(x, y, label='GeN-FOAM ' + columns[7])
plt.plot(zList, epsilon, label='DFM ' + columns[7])
# Add labels and legend
plt.xlabel(columns[0])
plt.ylabel('Void Fraction')
plt.legend()

# Show the plot
plt.show()
