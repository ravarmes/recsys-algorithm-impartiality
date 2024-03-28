import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the provided tables
data = {
    'MovieLens ALS': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -70.93, -71.80, -72.11, -72.59, -72.05], 'RMSE': [0, 0.49, 0.47, 0.49, 0.54, 0.51]},
    'MovieLens KNN': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -60.16, -79.55, -75.58, -79.28, -81.59], 'RMSE': [0, 4.20, 4.17, 2.06, 4.17, 3.73]},
    'GoodBooks ALS': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -35.00, -57.69, -70.23, -82.13, -82.23], 'RMSE': [0, 0.68, 1.15, 1.80, 1.91, 1.50]},
    'GoodBooks KNN': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -17.79, -21.36, -25.03, -27.37, -27.01], 'RMSE': [0, 1.89, 0.99, 2.03, 2.06, 2.92]},
    'Songs ALS': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -83.84, -96.60, -97.04, -96.37, -95.92], 'RMSE': [0, 5.26, 5.86, 6.24, 6.38, 6.71]},
    'Songs KNN': {'h': [0, 3, 5, 10, 15, 20], 'Rgrp': [0, -80.61, -83.41, -83.11, -82.06, -81.13], 'RMSE': [0, 3.03, 3.07, 2.91, 3.08, 3.51]}
}

fig, axs = plt.subplots(3, 2, figsize=(15, 13))  # Adjusting the size of the figure
fig.suptitle('Rgrp and RMSE Comparison by 95-5 Strategy, Algorithm, and Dataset')

# Iterating through the data to create the plots
for i, (title, data) in enumerate(data.items()):
    ax = axs.flat[i]
    ax.plot(data['h'], data['Rgrp'], marker='o', linestyle='-', color='tab:blue', label='Rgrp')
    ax.plot(data['h'], data['RMSE'], marker='s', linestyle='--', color='tab:orange', label='RMSE')

    # Using annotation to simulate a title in the middle of the plot
    ax.text(0.5, 0.5, title, ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Setting the x-axis ticks to only the specified integer values
    ax.set_xticks(data['h'])

    # Hiding the x-axis labels for the first four plots
    if i < 4:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    else:
        ax.set_xlabel('Number of Matrices (h)')
    ax.set_ylabel('Percentage Change (%)')
    ax.grid(True)
    ax.legend(loc='lower left')

    # Adding the point (0, 0) if necessary
    ax.scatter(0, 0, color='red')  # Red dot at (0, 0)

    # Adjusting the y-axis and x-axis scales
    ax.set_ylim([-100, 10])  # y-axis scale from 10 to -100
    ax.set_xlim([min(data['h']) - 1, max(data['h']) + 1])  # Space before the smallest h value and after the largest
    ax.set_yticks(np.arange(-100, 11, 10))  # From -100 to 10, in steps of 10

# Adjusting the spacing between the subplots
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.8, wspace=0.5)

# Final adjustments to improve layout and visualization
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
