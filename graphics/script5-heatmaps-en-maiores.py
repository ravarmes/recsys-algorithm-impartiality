import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que estes sejam os valores médios calculados previamente para cada combinação de dataset e algoritmo
# Para R_{grp}
medias_rgrp = np.array([
    [72.59, 97.04, 82.23],  # Médias de ALS para cada dataset
    [81.59, 83.41, 27.37]   # Médias de KNN para cada dataset
])

# Para RMSE
medias_rmse = np.array([
    [0.54, 6.71, 1.91],  # Médias de ALS para cada dataset
    [4.20, 3.51, 2.92]   # Médias de KNN para cada dataset
])

# Definindo os labels para os eixos
algorithms = ['ALS', 'KNN']
datasets = ['MovieLens', 'Songs', 'GoodBooks']

# Criando os heatmaps
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap para R_{grp}
sns.heatmap(medias_rgrp, ax=axs[0], annot=True, fmt=".2f", cmap="coolwarm", xticklabels=datasets, yticklabels=algorithms)
axs[0].set_title('Average Group Unfairness Reduction')
axs[0].set_xlabel('Dataset')
axs[0].set_ylabel('Algorithm')

# Heatmap para RMSE
sns.heatmap(medias_rmse, ax=axs[1], annot=True, fmt=".2f", cmap="coolwarm", xticklabels=datasets, yticklabels=algorithms)
axs[1].set_title('Average RMSE Increase')
axs[1].set_xlabel('Dataset')
axs[1].set_ylabel('Algorithm')

plt.tight_layout()
plt.show()
