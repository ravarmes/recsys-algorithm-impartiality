import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['MovieLens', 'Songs', 'GoodBooks']
algorithms = ['ALS', 'KNN']

# Data for group injustice reduction
data_group_injustice = [
    [72.59, 97.04, 82.23],  # ALS
    [81.59, 83.41, 27.37]   # KNN
]

# Data for RMSE increase
data_rmse_increase = [
    [0.54, 6.71, 1.91],  # ALS
    [4.20, 3.51, 2.92]   # KNN
]

# Angles for radar charts
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

# First radar chart: Algorithms by Datasets for Group Injustice Reduction
for data, algorithm in zip(data_group_injustice, algorithms):
    axs[0].plot(angles, data + data[:1], label=algorithm)
    axs[0].fill(angles, data + data[:1], alpha=0.1)
axs[0].set_xticks(angles[:-1])
axs[0].set_xticklabels(categories)
axs[0].set_title('Maximum Group Unfairness Reduction by Algorithms')
axs[0].legend()

# Second radar chart: Algorithms by Datasets for RMSE Increase
for data, algorithm in zip(data_rmse_increase, algorithms):
    axs[1].plot(angles, data + data[:1], label=algorithm)
    axs[1].fill(angles, data + data[:1], alpha=0.1)
axs[1].set_xticks(angles[:-1])
axs[1].set_xticklabels(categories)
axs[1].set_title('Maximum RMSE Increase by Algorithms')
axs[1].legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
