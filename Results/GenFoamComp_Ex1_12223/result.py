import numpy as np
import matplotlib.pyplot as plt

# Données pour les différents modèles
models = ['EPRIvoidModel', 'GEramp', 'modBestion', 'HEM1']
mean_void_fractions = [
    [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00178239, 0.01840426, 
     0.00089654, 0.01653151, 0.02096933, 0.02053501, 0.01706364, 0.01353465, 0.01173171],
    [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00178239, 0.01840426, 
     0.01829045, 0.00877975, 0.02504844, 0.03408152, 0.03823353, 0.04097126, 0.04447381],
    [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00178239, 0.01840426, 
     0.01802344, 0.00933612, 0.0255491, 0.03430187, 0.03802998, 0.04025491, 0.04319298],
    [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00178239, 0.01840426, 
     0.01387862, 0.02256264, 0.04653992, 0.06201385, 0.07162314, 0.07904273, 0.08660685]
]

# Erreurs moyennes, min et max pour les différents modèles
void_fraction_errors = {
    'mean': [0.006444686500815942, 0.012161337476910429, 0.012098687370677316, 0.021234443482673164],
    'min': [0.0001, 0.0001, 0.0001, 0.0001],
    'max': [0.020969333292636325, 0.04447380532560069, 0.04319298255650028, 0.08660685131074425]
}

# Création du graphique en barres pour les erreurs de la fraction de vide
x = np.arange(len(models))  # positions des modèles
width = 0.35  # largeur des barres

fig, ax = plt.subplots()

# Barres pour les erreurs moyennes de la fraction de vide
bars = ax.bar(x, void_fraction_errors['mean'], width, 
              yerr=[void_fraction_errors['mean'], [void_fraction_errors['max'][i] - void_fraction_errors['mean'][i] for i in range(len(models))]],
              capsize=5, color='skyblue', label='Erreur fraction de vide')

# Ajout des labels et titre
ax.set_xlabel('Modèles')
ax.set_ylabel('Erreurs de fraction de vide')
ax.set_title('Erreurs moyennes, min et max pour les modèles de fraction de vide')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Affichage du graphique
plt.tight_layout()
plt.show()