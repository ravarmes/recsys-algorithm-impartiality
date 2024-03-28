import matplotlib.pyplot as plt
import numpy as np

# Dados de Rgrp e RMSE para cada algoritmo e dataset
rgrp_als = [
    0.000008842, 0.000008578, 0.000008483, 0.000008338, 0.000008502,  # MovieLens
    0.000276807, 0.000058313, 0.000050649, 0.000062162, 0.000069824,  # Songs
    0.000224587, 0.000146169, 0.000102859, 0.000061734, 0.000061405   # GoodBooks
]

rgrp_knn = [
    0.000001200, 0.000000616, 0.000000736, 0.000000624, 0.000000555,  # MovieLens
    0.000000489, 0.000000418, 0.000000426, 0.000000452, 0.000000476,  # Songs
    0.000081283, 0.000077750, 0.000074121, 0.000071814, 0.000072164   # GoodBooks
]

rmse_als = [
    0.890821795, 0.890640562, 0.890861685, 0.891311053, 0.891029760,  # MovieLens
    0.741968918, 0.746204973, 0.748920480, 0.749877643, 0.752209714,  # Songs
    0.815134287, 0.818954503, 0.824180214, 0.825082121, 0.821748151   # GoodBooks
]

rmse_knn = [
    0.762339873, 0.762144990, 0.746680594, 0.762123255, 0.758920063,  # MovieLens
    0.274435523, 0.274535220, 0.274106664, 0.274573152, 0.275721166,  # Songs
    0.356572213, 0.353435120, 0.357077538, 0.357172978, 0.360204002   # GoodBooks
]

# Organizando os dados para boxplot
dados_rgrp = [rgrp_als, rgrp_knn]
dados_rmse = [rmse_als, rmse_knn]

# Criando os boxplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot para Rgrp
axs[0].boxplot(dados_rgrp, labels=['ALS', 'KNN'])
axs[0].set_title('Group Unfairness Reduction')
axs[0].set_ylabel('Group Unfairness')
axs[0].set_xlabel('Algorithm')

# Boxplot para RMSE
axs[1].boxplot(dados_rmse, labels=['ALS', 'KNN'])
axs[1].set_title('RMSE Variation')
axs[1].set_ylabel('RMSE')
axs[1].set_xlabel('Algorithm')

plt.tight_layout()
plt.show()
