import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calculando agora para todos os casos
medias_rgrp = np.array([
    [np.mean([70.93, 71.8, 72.11, 72.59, 72.05]), np.mean([35.00, 57.69, 70.23, 82.13, 82.23]), np.mean([83.84, 96.6, 97.04, 96.37, 95.92])],  # ALS
    [np.mean([60.16, 79.55, 75.58, 79.28, 81.59]), np.mean([17.79, 21.36, 25.03, 27.37, 27.01]), np.mean([80.61, 83.41, 83.11, 82.06, 81.13])]   # KNN
])

medias_rmse = np.array([
    [np.mean([0.49, 0.47, 0.49, 0.54, 0.51]), np.mean([0.68, 1.15, 1.80, 1.91, 1.50]), np.mean([5.26, 5.86, 6.24, 6.38, 6.71])],  # ALS
    [np.mean([4.20, 4.17, 2.06, 4.17, 3.73]), np.mean([1.89, 0.99, 2.03, 2.06, 2.92]), np.mean([3.03, 3.07, 2.91, 3.08, 3.51])]   # KNN
])

# Definindo os labels para os eixos
algorithms = ['ALS', 'KNN']
datasets = ['MovieLens', 'GoodBooks', 'Songs']

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
