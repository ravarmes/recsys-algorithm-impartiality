import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the provided tables
data = {
    'RecSysALS NR': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54]},
    'RecSysALS Gender': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -2.6, -5.14, -11.06, -13.57], 'RMSE': [0, 4.20, 4.17, 2.06, 4.17]},
    'RecSysALS Age': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -35.00, -57.69, -70.23, -82.13], 'RMSE': [0, 0.68, 1.15, 1.80, 1.91]},
    'RecSysKNN NR': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54]},
    'RecSysKNN Gender': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -50.83, -65.60, -77.52, -83.06], 'RMSE': [0, 4.20, 4.17, 2.06, 4.17]},
    'RecSysKNN Age': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -35.00, -57.69, -70.23, -82.13], 'RMSE': [0, 0.68, 1.15, 1.80, 1.91]},
    'RecSysNMF NR': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54]},
    'RecSysNMF Gender': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -1.15, -5.46, -6.82, -12.38], 'RMSE': [0, 4.20, 4.17, 2.06, 4.17]},
    'RecSysNMF Age': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -35.00, -57.69, -70.23, -82.13], 'RMSE': [0, 0.68, 1.15, 1.80, 1.91]},
    'RecSysNCF NR': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54]},
    'RecSysNCF Gender': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -15.45, -22.38, -23.75, -16.21], 'RMSE': [0, 4.20, 4.17, 2.06, 4.17]},
    'RecSysNCF Age': {'h': [0, 3, 5, 10, 15], 'Rgrp': [0, -35.00, -57.69, -70.23, -82.13], 'RMSE': [0, 0.68, 1.15, 1.80, 1.91]}
}


fig, axs = plt.subplots(4, 3, figsize=(15, 18))  # Adjusting the size and shape of the figure
fig.suptitle('Rgrp and RMSE Comparison by Dataset MovieLens, Algorithm, and Grouping')

# Iterating through the data to create the plots
for i, (title, data_dict) in enumerate(data.items()):
    ax = axs.flat[i]
    ax.plot(data_dict['h'], data_dict['Rgrp'], marker='o', linestyle='-', color='tab:blue', label='Rgrp')
    ax.plot(data_dict['h'], data_dict['RMSE'], marker='s', linestyle='--', color='tab:orange', label='RMSE')

    ax.text(0.5, 0.5, title, ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xticks(data_dict['h'])
    
    # Hiding the x-axis labels for all but the last row of plots
    if i < 9:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    else:
        ax.set_xlabel('Number of Matrices (h)')
    ax.set_ylabel('Percentage Change (%)')
    ax.grid(True)
    ax.legend(loc='lower left')

    ax.scatter(0, 0, color='red')  # Red dot at (0, 0)

    ax.set_ylim([-100, 10])  # y-axis scale from 10 to -100
    ax.set_xlim([min(data_dict['h']) - 1, max(data_dict['h']) + 1])  # Space before and after the h values
    ax.set_yticks(np.arange(-100, 11, 10))  # Y-axis ticks from -100 to 10, in steps of 10

# Adjusting the spacing between the subplots
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.4)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


