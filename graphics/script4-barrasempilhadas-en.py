import matplotlib.pyplot as plt
import numpy as np

# Médias calculadas previamente
# Médias de Rgrp e RMSE para ALS e KNN em cada dataset
medias_rgrp_als = [np.mean([72.59, 97.04, 82.23]), np.mean([81.59, 83.41, 27.37])]
medias_rmse_als = [np.mean([0.54, 6.71, 1.91]), np.mean([4.20, 3.51, 2.92])]

datasets = ['MovieLens', 'Songs', 'GoodBooks']
algoritmos = ['ALS', 'KNN']

# Calculando posições das barras
barWidth = 0.35
r1 = np.arange(len(medias_rgrp_als))
r2 = [x + barWidth for x in r1]

# Criando gráfico de barras
plt.figure(figsize=(10, 6))

# Desenhando barras para Rgrp
plt.bar(r1, medias_rgrp_als, color='blue', width=barWidth, edgecolor='grey', label='R_{grp}')
# Desenhando barras para RMSE
plt.bar(r2, medias_rmse_als, color='orange', width=barWidth, edgecolor='grey', label='RMSE')

# Adicionando legendas, título e customizando eixos
plt.xlabel('Algorithm', fontweight='bold')
plt.xticks([r + barWidth/2 for r in range(len(medias_rgrp_als))], algoritmos)
plt.ylabel('Metric Value')
plt.title('Comparison of R_{grp} and RMSE by Algorithm')
plt.legend()

plt.tight_layout()
plt.show()
