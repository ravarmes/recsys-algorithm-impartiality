import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Dados - médias das variações percentuais de Rgrp e RMSE para cada configuração
labels=np.array(['MovieLens Hierárquico', 'Songs Hierárquico', 'MovieLens 95-5', 'Songs 95-5'])
Rgrp_means = [-20.47, -35.12, -71.70, -90.81]  # Média das variações de Rgrp
RMSE_means = [0.30, -4.58, 0.50, 0.67]  # Média das variações de RMSE

# Número de variáveis
num_vars = len(labels)

# Ângulos para cada eixo no gráfico de radar
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# O gráfico é circular, então precisamos "fechar" o círculo:
Rgrp_means += Rgrp_means[:1]
RMSE_means += RMSE_means[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, Rgrp_means, color='red', alpha=0.25)
ax.plot(angles, Rgrp_means, color='red', label='Média Rgrp (%)')
ax.fill(angles, RMSE_means, color='blue', alpha=0.25)
ax.plot(angles, RMSE_means, color='blue', label='Média RMSE (%)')

# Etiquetas para cada configuração
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.show()
