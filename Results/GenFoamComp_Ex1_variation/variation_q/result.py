import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

printing = False
print_ecart_puissance = False
ecart_correl = False
plotting_power = True

if printing:
    # Read the Excel file
    df = pd.read_excel(r'C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\Results\GenFoamComp_Ex1_variation\variation_q\merge_VF.xlsx')

    # Create empty lists for each column
    columns = df.columns.tolist()
    data = [[] for _ in columns]

    # Iterate over each row and append values to the corresponding list
    for index, row in df.iterrows():
        for i, col in enumerate(columns):
            data[i].append(row[col])

    cas = 4

    xGenFoam = data[0]
    #12 14 16 18 20 22 50 80 85
    genfoamCASE = [data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16], data[18]]

    df = pd.read_excel(r'C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\Results\GenFoamComp_Ex1_variation\variation_q\merge_VF.xlsx', sheet_name=cas+1)

    # Create empty lists for each column
    columns = df.columns.tolist()
    print(columns)
    data = [[] for _ in columns]

    # Iterate over each row and append values to the corresponding list
    for index, row in df.iterrows():
        for i, col in enumerate(columns):
            data[i].append(row[col])

    xDFM = data[0]
    Case12 = [data[1], data[2], data[3], data[4]]


    fig1, ax1 = plt.subplots()
    for i in range(len(Case12)):
        ax1.plot(xDFM, Case12[i], label=columns[i+1])
    ax1.plot(xGenFoam, genfoamCASE[cas], label="GenFoam")
    ax1.set_xlabel("Axial position in m")
    ax1.set_ylabel("Void fraction")
    ax1.set_title("Void fraction at 20e8 W/m^3")
    ax1.legend(loc="best")
    plt.show()

    # Créer un DataFrame pour la première série
    df1 = pd.DataFrame({'x': xGenFoam, 'y': genfoamCASE[cas]})

    # Créer un DataFrame pour la deuxième série
    df2 = pd.DataFrame({'x': xDFM, 'y': Case12[0]})
    df3 = pd.DataFrame({'x': xDFM, 'y': Case12[1]})
    df4 = pd.DataFrame({'x': xDFM, 'y': Case12[2]})
    df5 = pd.DataFrame({'x': xDFM, 'y': Case12[3]})

    # Interpolation de la deuxième série sur les abscisses de la première série
    df2_interp = pd.DataFrame({'x': df1['x']})
    df2_interp['y_interp'] = np.interp(df1['x'], df2['x'], df2['y'])

    df3_interp = pd.DataFrame({'x': df1['x']})
    df3_interp['y_interp'] = np.interp(df1['x'], df3['x'], df3['y'])

    df4_interp = pd.DataFrame({'x': df1['x']})
    df4_interp['y_interp'] = np.interp(df1['x'], df4['x'], df4['y'])

    df5_interp = pd.DataFrame({'x': df1['x']})
    df5_interp['y_interp'] = np.interp(df1['x'], df5['x'], df5['y'])

    # Calcul de la différence entre les deux séries
    ecart0 = abs(df1['y'] - df2_interp['y_interp'])
    ecart1 = abs(df1['y'] - df3_interp['y_interp'])
    ecart2 = abs(df1['y'] - df4_interp['y_interp'])
    ecart3 = abs(df1['y'] - df5_interp['y_interp'])

    # Affichage de la différence
    fig, ax = plt.subplots()
    ax.plot(df1['x'], ecart0, label=f"Ecart {columns[1]}.")
    ax.plot(df1['x'], ecart1, label=f"Ecart {columns[2]}.")
    ax.plot(df1['x'], ecart2, label=f"Ecart {columns[3]}.")
    ax.plot(df1['x'], ecart3, label=f"Ecart {columns[4]}.")
    ax.set_xlabel("x")
    ax.set_ylabel("Ecart")
    ax.set_title("Ecart avec GenFoam")
    ax.legend(loc="best")
    plt.show()

    print(f"Max ecart {columns[1]}: ecart max {max(ecart0)}, ecart moyen {np.mean(ecart0)}, ecart min {min(ecart0)}")
    print(f"Max ecart {columns[2]}: ecart max {max(ecart1)}, ecart moyen {np.mean(ecart1)}, ecart min {min(ecart1)}")
    print(f"Max ecart {columns[3]}: ecart max {max(ecart2)}, ecart moyen {np.mean(ecart2)}, ecart min {min(ecart2)}")
    print(f"Max ecart {columns[4]}: ecart max {max(ecart3)}, ecart moyen {np.mean(ecart3)}, ecart min {min(ecart3)}")


if print_ecart_puissance:
    # Données
    puissance = [12e8, 14e8, 16e8, 18e8, 20e8, 22e8, ]

    # Ecart max pour chaque méthode
    ecart_max_epri = [0.007957565645330471, 0.03483594937280719, 0.042445060551209356, 0.03428081867009687, 0.04752017959161788, 0.05340659039282819]
    ecart_max_geramp = [0.007957565645330471, 0.03483594937280719, 0.042445060551209356, 0.03894832486936117, 0.04855506133385288, 0.05503712729811503]
    ecart_max_modbestion = [0.007957565645330471, 0.03483594937280719, 0.042445060551209356, 0.03867546475847897, 0.04848843675316924, 0.054969772498235196]
    ecart_max_hem = [0.007957565645330471, 0.03483594937280719, 0.042445060551209356, 0.038470107743786064, 0.05521552644397154, 0.07164823434330692]

    # Ecart moyen pour chaque méthode
    ecart_moyen_epri = [0.0003043594130353338, 0.0026034886713439264, 0.004880269901357892, 0.007353864216061566, 0.010746427163824625, 0.013240535715930897]
    ecart_moyen_geramp = [0.0003043594130353338, 0.0035024284632012423, 0.0054233999640328386, 0.006408251140745191, 0.008130264890761433, 0.01030659605141135]
    ecart_moyen_modbestion = [0.0003043594130353338, 0.003470835099733863, 0.005340574869076066, 0.006318040332568281, 0.007964386063706518, 0.009993362724252473]
    ecart_moyen_hem = [0.0003043594130353338, 0.0029525919269220244, 0.005459503676647627, 0.008493819281389377, 0.012957015282821756, 0.017999001668950642]

    # Tracé des écarts max
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(puissance, ecart_max_epri, marker='o', label="EPRIvoidModel")
    plt.plot(puissance, ecart_max_geramp, marker='o', label="GEramp")
    plt.plot(puissance, ecart_max_modbestion, marker='o', label="modBestion")
    plt.plot(puissance, ecart_max_hem, marker='o', label="HEM1")
    plt.xlabel('Puissance (W)')
    plt.ylabel('Ecart Max')
    plt.title("Ecart Max en fonction de la Puissance")
    plt.legend()

    # Tracé des écarts moyens
    plt.subplot(1, 2, 2)
    plt.plot(puissance, ecart_moyen_epri, marker='o', label="EPRIvoidModel")
    plt.plot(puissance, ecart_moyen_geramp, marker='o', label="GEramp")
    plt.plot(puissance, ecart_moyen_modbestion, marker='o', label="modBestion")
    plt.plot(puissance, ecart_moyen_hem, marker='o', label="HEM1")
    plt.xlabel('Puissance (W)')
    plt.ylabel('Ecart Moyen')
    plt.title("Ecart Moyen en fonction de la Puissance")
    plt.legend()

    plt.tight_layout()
    plt.show()

if ecart_correl:
    # Données
    x_values = np.arange(1, 20)

    # Blasius Correlation
    mean_void_Blasius = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 
                        0.00178239, 0.01840426, 0.00089962, 0.01690415, 0.02130617, 0.02078514, 0.01722237, 0.01360028, 0.01170541]
    void_error_Blasius = 0.0065057795945787115
    void_max_error_Blasius = 0.02130617119897872
    temp_error_Blasius = 0.18491259400801818
    temp_max_error_Blasius = 0.40512525557211704
    pressure_error_Blasius = 0.11667265645176161
    pressure_max_error_Blasius = 0.13926098235912138
    velocity_error_Blasius = 5.900214520593739
    velocity_max_error_Blasius = 9.73798135289633

    # Churchill Correlation
    mean_void_churchill = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                        0.00178239, 0.01840426, 0.00053779, 0.01708538, 0.0214424, 0.02086249, 0.01724692, 0.01357785, 0.01164169]
    void_error_churchill = 0.006504272132589755
    void_max_error_churchill = 0.021442399745236027
    temp_error_churchill = 0.18441305009712136
    temp_max_error_churchill = 0.40461195424153656
    pressure_error_churchill = 0.13737893050235309
    pressure_max_error_churchill = 0.1555397469754948
    velocity_error_churchill = 5.884595698791065
    velocity_max_error_churchill = 9.72434324096621

    # Création des graphiques
    plt.figure(figsize=(12, 8))

    # Tracé de la mean voidFraction
    plt.subplot(2, 1, 1)
    plt.plot(x_values, mean_void_Blasius, marker='o', label="Blasius")
    plt.plot(x_values, mean_void_churchill, marker='o', label="Churchill")
    plt.xlabel('Index')
    plt.ylabel('Mean Void Fraction')
    plt.title('Mean Void Fraction for Blasius and Churchill Correlations')
    plt.legend()

    # Tracé des erreurs moyennes et maximales
    labels = ['Void Fraction', 'Temperature (%)', 'Pressure (%)', 'Velocity (%)']
    Blasius_errors = [void_error_Blasius, temp_error_Blasius, pressure_error_Blasius, velocity_error_Blasius]
    Blasius_max_errors = [void_max_error_Blasius, temp_max_error_Blasius, pressure_max_error_Blasius, velocity_max_error_Blasius]
    churchill_errors = [void_error_churchill, temp_error_churchill, pressure_error_churchill, velocity_error_churchill]
    churchill_max_errors = [void_max_error_churchill, temp_max_error_churchill, pressure_max_error_churchill, velocity_max_error_churchill]

    x = np.arange(len(labels))  # position des barres
    width = 0.35  # largeur des barres

    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, Blasius_errors, width, label='Blasius (Mean)', color='lightblue')
    plt.bar(x + width/2, Blasius_max_errors, width, label='Blasius (Max)', color='')
    plt.bar(x - width/2, churchill_errors, width, bottom=Blasius_errors, label='Churchill (Mean)', color='r')
    plt.bar(x + width/2, churchill_max_errors, width, bottom=Blasius_max_errors, label='Churchill (Max)', color='m')

    plt.ylabel('Error')
    plt.title('Mean and Maximum Errors for Blasius and Churchill Correlations')
    plt.xticks(x, labels)
    plt.legend()

    plt.tight_layout()
    plt.show()



if plotting_power:
    # Read the Excel file
    df = pd.read_excel(r'C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\Results\GenFoamComp_Ex1_variation\variation_q\merge_VF.xlsx')

    # Create empty lists for each column
    columns = df.columns.tolist()
    data = [[] for _ in columns]

    # Iterate over each row and append values to the corresponding list
    for index, row in df.iterrows():
        for i, col in enumerate(columns):
            data[i].append(row[col])

    ecart_puissance = {
    }    

    xGenFoam = data[0]
    puissance = [12e8, 14e8, 16e8, 18e8, 20e8, 22e8, 50e8, 80e8, 85e8]
    genfoamCASE = [data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16], data[18]]
    
    for cas in range(len(genfoamCASE)):

        df = pd.read_excel(r'C:\Users\cleme\OneDrive\Documents\Poly\BWR\driftFluxModel\Results\GenFoamComp_Ex1_variation\variation_q\merge_VF.xlsx', sheet_name=cas+1)

        # Create empty lists for each column
        columns = df.columns.tolist()
        print(columns)
        data = [[] for _ in columns]

        # Iterate over each row and append values to the corresponding list
        for index, row in df.iterrows():
            for i, col in enumerate(columns):
                data[i].append(row[col])

        xDFM = data[0]
        Case12 = [data[1], data[2], data[3], data[4]]


        """ fig1, ax1 = plt.subplots()
        for i in range(len(Case12)):
            ax1.plot(xDFM, Case12[i], label=columns[i+1])
        ax1.plot(xGenFoam, genfoamCASE[cas], label="GenFoam")
        ax1.set_xlabel("Axial position in m")
        ax1.set_ylabel("Void fraction")
        ax1.set_title("Void fraction at 20e8 W/m^3")
        ax1.legend(loc="best")
        plt.show() """

        # Créer un DataFrame pour la première série
        df1 = pd.DataFrame({'x': xGenFoam, 'y': genfoamCASE[cas]})

        # Créer un DataFrame pour la deuxième série
        df2 = pd.DataFrame({'x': xDFM, 'y': Case12[0]})
        df3 = pd.DataFrame({'x': xDFM, 'y': Case12[1]})
        df4 = pd.DataFrame({'x': xDFM, 'y': Case12[2]})
        df5 = pd.DataFrame({'x': xDFM, 'y': Case12[3]})

        # Interpolation de la deuxième série sur les abscisses de la première série
        df2_interp = pd.DataFrame({'x': df1['x']})
        df2_interp['y_interp'] = np.interp(df1['x'], df2['x'], df2['y'])

        df3_interp = pd.DataFrame({'x': df1['x']})
        df3_interp['y_interp'] = np.interp(df1['x'], df3['x'], df3['y'])

        df4_interp = pd.DataFrame({'x': df1['x']})
        df4_interp['y_interp'] = np.interp(df1['x'], df4['x'], df4['y'])

        df5_interp = pd.DataFrame({'x': df1['x']})
        df5_interp['y_interp'] = np.interp(df1['x'], df5['x'], df5['y'])

        # Calcul de la différence entre les deux séries
        ecart0 = abs(df1['y'] - df2_interp['y_interp'])
        ecart1 = abs(df1['y'] - df3_interp['y_interp'])
        ecart2 = abs(df1['y'] - df4_interp['y_interp'])
        ecart3 = abs(df1['y'] - df5_interp['y_interp'])

        """ # Affichage de la différence
        fig, ax = plt.subplots()
        ax.plot(df1['x'], ecart0, label=f"Ecart {columns[1]}.")
        ax.plot(df1['x'], ecart1, label=f"Ecart {columns[2]}.")
        ax.plot(df1['x'], ecart2, label=f"Ecart {columns[3]}.")
        ax.plot(df1['x'], ecart3, label=f"Ecart {columns[4]}.")
        ax.set_xlabel("x")
        ax.set_ylabel("Ecart")
        ax.set_title("Ecart avec GenFoam")
        ax.legend(loc="best")
        plt.show() """

        #print(f"Max ecart {columns[1]}: ecart max {max(ecart0)}, ecart moyen {np.mean(ecart0)}, ecart min {min(ecart0)}")
        #print(f"Max ecart {columns[2]}: ecart max {max(ecart1)}, ecart moyen {np.mean(ecart1)}, ecart min {min(ecart1)}")
        #print(f"Max ecart {columns[3]}: ecart max {max(ecart2)}, ecart moyen {np.mean(ecart2)}, ecart min {min(ecart2)}")
        #print(f"Max ecart {columns[4]}: ecart max {max(ecart3)}, ecart moyen {np.mean(ecart3)}, ecart min {min(ecart3)}")

        ecart = {
            columns[1]: [max(ecart0), np.mean(ecart0), min(ecart0)],
            columns[2]: [max(ecart1), np.mean(ecart1), min(ecart1)],
            columns[3]: [max(ecart2), np.mean(ecart2), min(ecart2)],
            columns[4]: [max(ecart3), np.mean(ecart3), min(ecart3)]
        }

        ecart_puissance[f'{puissance[cas]}'] = ecart


    ecart0=[]
    ecart1=[]
    ecart2=[]
    ecart3=[]
    for key in ecart_puissance.keys():
        ecart0.append(ecart_puissance[key][columns[1]][1])
        ecart1.append(ecart_puissance[key][columns[2]][1])
        ecart2.append(ecart_puissance[key][columns[3]][1])
        ecart3.append(ecart_puissance[key][columns[4]][1])

    print(f'ecart0: {ecart0}')
    print(f'ecart1: {ecart1}')
    print(f'ecart2: {ecart2}')
    print(f'ecart3: {ecart3}')

    # Affichage de la différence
    fig, ax = plt.subplots()
    ax.plot(puissance, ecart0, label=f"Ecart {columns[1]}.")
    ax.plot(puissance, ecart1, label=f"Ecart {columns[2]}.")
    ax.plot(puissance, ecart2, label=f"Ecart {columns[3]}.")
    ax.plot(puissance, ecart3, label=f"Ecart {columns[4]}.")
    ax.set_xlabel("x")
    ax.set_ylabel("Ecart")
    ax.set_title("Ecart avec GenFoam")
    ax.legend(loc="best")
    plt.show()
    