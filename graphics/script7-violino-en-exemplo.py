import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Supondo que tenhamos coletado os seguintes dados de R_{grp} e RMSE para cada algoritmo e dataset
rgrp_data = {
    'MovieLens': {
        'ALS': np.random.normal(loc=0.001, scale=0.0005, size=100),
        'KNN': np.random.normal(loc=0.002, scale=0.0005, size=100)
    },
    'Songs': {
        'ALS': np.random.normal(loc=0.003, scale=0.0005, size=100),
        'KNN': np.random.normal(loc=0.004, scale=0.0005, size=100)
    },
    'GoodBooks': {
        'ALS': np.random.normal(loc=0.0025, scale=0.0005, size=100),
        'KNN': np.random.normal(loc=0.0015, scale=0.0005, size=100)
    }
}

rmse_data = {
    'MovieLens': {
        'ALS': np.random.normal(loc=0.9, scale=0.1, size=100),
        'KNN': np.random.normal(loc=0.8, scale=0.1, size=100)
    },
    'Songs': {
        'ALS': np.random.normal(loc=0.7, scale=0.1, size=100),
        'KNN': np.random.normal(loc=0.6, scale=0.1, size=100)
    },
    'GoodBooks': {
        'ALS': np.random.normal(loc=0.85, scale=0.1, size=100),
        'KNN': np.random.normal(loc=0.75, scale=0.1, size=100)
    }
}

# Preparando os dados para plotagem
rgrp_values, rgrp_labels, rmse_values, rmse_labels = [], [], [], []

for dataset in rgrp_data:
    for algorithm in rgrp_data[dataset]:
        rgrp_values.extend(rgrp_data[dataset][algorithm])
        rgrp_labels.extend([f"{algorithm} - {dataset}"] * len(rgrp_data[dataset][algorithm]))
        
for dataset in rmse_data:
    for algorithm in rmse_data[dataset]:
        rmse_values.extend(rmse_data[dataset][algorithm])
        rmse_labels.extend([f"{algorithm} - {dataset}"] * len(rmse_data[dataset][algorithm]))

# Criando gráficos de violino
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de violino para R_{grp}
sns.violinplot(x=rgrp_labels, y=rgrp_values, ax=axs[0])
axs[0].set_title('R_{grp} Distribution by Algorithm and Dataset')
axs[0].set_xlabel('Algorithm - Dataset')
axs[0].set_ylabel('R_{grp}')
axs[0].tick_params(axis='x', rotation=45)

# Gráfico de violino para RMSE
sns.violinplot(x=rmse_labels, y=rmse_values, ax=axs[1])
axs[1].set_title('RMSE Distribution by Algorithm and Dataset')
axs[1].set_xlabel('Algorithm - Dataset')
axs[1].set_ylabel('RMSE')
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
