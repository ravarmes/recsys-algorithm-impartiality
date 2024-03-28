import matplotlib.pyplot as plt
import numpy as np

# Exemplo de dados - substitua pelos seus dados reais
dados = {
    'Hierarchical Strategy - MovieLens dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -17.32, -19.00, -20.07, -22.90, -23.07], 'RMSE': [0, 0.42, 0.35, 0.30, 0.22, 0.19]},
    'Hierarchical Strategy - Songs dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -22.06, -30.20, -37.92, -41.64, -43.68], 'RMSE': [0, -3.99, -4.27, -4.73, -4.88, -5.02]},
    '95-5 Strategy - MovieLens dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59, -72.05], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54, 0.51]},
    '95-5 Strategy - Songs dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -84.80, -93.86, -93.87, -93.49, -93.11], 'RMSE': [0, 0.13, 0.08, 1.18, 0.81, 1.16]}
}

fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
fig.suptitle('Comparison of Rgrp and RMSE by Cluster and Dataset Configuration')

# Ajuste para cada subplot
for i, (title, data) in enumerate(dados.items()):
    ax = axs.flat[i]
    ax.plot(data['h'], data['Rgrp'], marker='o', linestyle='-', color='tab:blue', label='Rgrp')
    ax.plot(data['h'], data['RMSE'], marker='s', linestyle='--', color='tab:orange', label='RMSE')
    ax.set_title(title)
    if i >= 2:  # Apenas para os subplots inferiores
        ax.set_xlabel('Number of Matrices (h)')
    ax.set_ylabel('Percentage Change (%)')
    ax.grid(True)
    ax.legend()

    # Ajustando as escalas do eixo Y e X
    ax.set_ylim([-100, 5])
    ax.set_yticks(np.arange(-100, 6, 10))  # De -100 a 5, de 10 em 10
    
# Ajuste dos ticks do eixo X para incluir o valor 3 e excluir o valor 2.5
plt.xticks(np.arange(3, 21, step=1))

# Ajustes finais para melhorar layout e visualização
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
