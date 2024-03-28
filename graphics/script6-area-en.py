import matplotlib.pyplot as plt
import numpy as np

# Supondo que estes sejam os dados acumulados ou variados de R_{grp} e RMSE para cada algoritmo ao longo dos datasets
datasets = ['MovieLens', 'Songs', 'GoodBooks']

# Dados fictícios de R_{grp} acumulados para cada algoritmo
rgrp_als = np.array([72, 97, 82])  # Acumulado ou variado ao longo dos datasets
rgrp_knn = np.array([81, 83, 27])  # Acumulado ou variado ao longo dos datasets

# Dados fictícios de RMSE acumulados para cada algoritmo
rmse_als = np.array([0.54, 6.71, 1.91])  # Acumulado ou variado ao longo dos datasets
rmse_knn = np.array([4.20, 3.51, 2.92])  # Acumulado ou variado ao longo dos datasets

# Criando gráficos de área
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico de área para R_{grp}
axs[0].stackplot(datasets, rgrp_als, rgrp_knn, labels=['ALS', 'KNN'], alpha=0.5)
axs[0].set_title('Accumulated Group Unfairness Reduction')
axs[0].set_xlabel('Dataset')
axs[0].set_ylabel('Accumulated R_{grp}')
axs[0].legend(loc='upper left')

# Gráfico de área para RMSE
axs[1].stackplot(datasets, rmse_als, rmse_knn, labels=['ALS', 'KNN'], alpha=0.5)
axs[1].set_title('Accumulated RMSE Increase')
axs[1].set_xlabel('Dataset')
axs[1].set_ylabel('Accumulated RMSE')
axs[1].legend(loc='upper left')

plt.tight_layout()
plt.show()
