import matplotlib.pyplot as plt
import numpy as np

# Dados extraídos das tabelas fornecidas
dados = {
    'MovieLens ALS': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59, -72.05], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54, 0.51]},
    'MovieLens KNN': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -60.16, -79.55, -75.58, -79.28, -81.59], 'RMSE': [0, 4.20, 4.17, 2.06, 4.17, 3.73]},
    'Songs ALS': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -83.84, -96.60, -97.04, -96.37, -95.92], 'RMSE': [0, 5.26, 5.86, 6.24, 6.38, 6.71]},
    'Songs KNN': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -80.61, -83.41, -83.11, -82.06, -81.13], 'RMSE': [0, 3.03, 3.07, 2.91, 3.08, 3.51]},
    'GoodBooks ALS': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -35.00, -57.69, -70.23, -82.13, -82.23], 'RMSE': [0, 0.68, 1.15, 1.80, 1.91, 1.50]},
    'GoodBooks KNN': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -17.79, -21.36, -25.03, -27.37, -27.01], 'RMSE': [0, 1.89, 0.99, 2.03, 2.06, 2.92]},
}

fig, axs = plt.subplots(3, 2, figsize=(15, 13))  # Ajuste do tamanho da figura
fig.suptitle('Comparação de Rgrp e RMSE por Estratégia 95-5, Algoritmo e Dataset')

# Iteração pelos dados para criar os gráficos
for i, (title, data) in enumerate(dados.items()):
    ax = axs.flat[i]
    ax.plot(data['h'], data['Rgrp'], marker='o', linestyle='-', color='tab:blue', label='Rgrp')
    ax.plot(data['h'], data['RMSE'], marker='s', linestyle='--', color='tab:orange', label='RMSE')

    # Usando anotação para simular um título no meio do gráfico
    ax.text(0.5, 0.5, title, ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Define os ticks do eixo X para serem apenas os valores inteiros especificados
    ax.set_xticks(data['h'])

    # Escondendo os rótulos do eixo X nos primeiros quatro gráficos
    if i < 4:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    else:
        ax.set_xlabel('Número de Matrizes (h)')

    ax.set_ylabel('Variação Percentual (%)')
    ax.grid(True)
    ax.legend(loc='lower left')

    # Adicionando o ponto (0, 0) se necessário
    ax.scatter(0, 0, color='red')  # Ponto vermelho em (0, 0)

    # Ajustando as escalas do eixo Y e X
    ax.set_ylim([-100, 10])  # Escala de y de 10 até -100
    ax.set_xlim([min(data['h']) - 1, max(data['h']) + 1])  # Espaço antes do menor valor de h e após o maior
    ax.set_yticks(np.arange(-100, 11, 10))  # De -100 a 10, de 10 em 10

# Ajustando o espaçamento entre os subplots
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.8, wspace=0.5)

# Ajustes finais para melhorar layout e visualização
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
