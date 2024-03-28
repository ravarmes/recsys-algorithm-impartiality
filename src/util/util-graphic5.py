import matplotlib.pyplot as plt
import numpy as np

data = {
    'Hierarchical Strategy - MovieLens dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -17.32, -19.00, -20.07, -22.90, -23.07], 'RMSE': [0, 0.42, 0.35, 0.30, 0.22, 0.19]},
    'Hierarchical Strategy - Songs dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -22.06, -30.20, -37.92, -41.64, -43.68], 'RMSE': [0, -3.99, -4.27, -4.73, -4.88, -5.02]},
    '95-5 Strategy - MovieLens dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59, -72.05], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54, 0.51]},
    '95-5 Strategy - Songs dataset': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -84.80, -93.86, -93.87, -93.49, -93.11], 'RMSE': [0, 0.13, 0.08, 1.18, 0.81, 1.16]}
}

strategies = list(data.keys())
h_index = 3  # Index for 'h=10', which is the fourth element in the 'h' list

# Preparing data for the plot
rgrp_values = [data[strategy]['Rgrp'][h_index] for strategy in strategies]
strategy_labels = [strategy.split(' - ')[0] for strategy in strategies]  # Simplifying names
dataset_labels = [strategy.split(' - ')[1] for strategy in strategies]

# Bar plot for 'Rgrp' at 'h=10'
fig, ax = plt.subplots(figsize=(10, 6))
bar_positions = np.arange(len(strategies))

ax.bar(bar_positions, rgrp_values, color='skyblue', label='Rgrp at h=10')
ax.set_xlabel('Strategy and Dataset')
ax.set_ylabel('Rgrp Change (%)')
ax.set_title('Comparison of Rgrp Change (%) at h=10 by Strategy and Dataset')
ax.set_xticks(bar_positions)
ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
ax.legend()

# Adding strategy labels above bars for clarity
for i, label in enumerate(strategy_labels):
    ax.text(i, rgrp_values[i], f'{label}\n{rgrp_values[i]:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
