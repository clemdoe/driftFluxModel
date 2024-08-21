import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from THM_DONJON_parser import THM_DONJON_parser

donjon = THM_DONJON_parser("BWR/driftFluxModel/THM_Comparaison/thm_genfoam_comp.result", 0, 0, 0, 20, 0.2)
print(donjon.TCOOL)

# Read the Excel file
df = pd.read_excel("BWR\driftFluxModel\THM_Comparaison\Firstopenfoam.xlsx")

# Create empty lists for each column
columns = df.columns.tolist()
data = [[] for _ in columns]

# Iterate over each row and append values to the corresponding list
for index, row in df.iterrows():
    for i, col in enumerate(columns):
        data[i].append(row[col])

z = np.linspace(0, 1.655, 20)
TCOOL = [donjon.TCOOL[0][-(i+1)] for i in range(len(donjon.TCOOL[0]))]
print(z, TCOOL)
plt.plot(z, TCOOL, label="THM")
plt.plot(data[0], data[3], label="GenFOAM")
plt.show()