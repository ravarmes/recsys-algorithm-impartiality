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

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Convert the second row to polar plots manually
axs[1, 0] = plt.subplot(2, 2, 3, polar=True)
axs[1, 1] = plt.subplot(2, 2, 4, polar=True)

# Radar charts for the first row
for i, data_list in enumerate([data_group_injustice, data_rmse_increase]):
    for data, algorithm in zip(data_list, algorithms):
        axs[0, i].plot(angles, data + data[:1], label=algorithm)
        axs[0, i].fill(angles, data + data[:1], alpha=0.1)
    axs[0, i].set_xticks(angles[:-1])
    axs[0, i].set_xticklabels(categories)
    axs[0, i].set_title(f'{"Group Unfairness Reduction" if i==0 else "RMSE Increase"} by Algorithms')
    axs[0, i].legend()

# Bar charts for the second row (Adapting the approach since polar=True is no longer applicable)
# We'll need to create standard cartesian subplots for bar charts instead.
fig.delaxes(axs[1][0])  # Remove the previously added polar subplot
fig.delaxes(axs[1][1])  # Remove the previously added polar subplot

# Adding new subplots for bar charts
axs[1, 0] = fig.add_subplot(2, 2, 3)
axs[1, 1] = fig.add_subplot(2, 2, 4)

# Variables for bar charts
bar_width = 0.35
index = np.arange(len(categories))

# Bar charts
for idx, data_list in enumerate([data_group_injustice, data_rmse_increase]):
    for i, data in enumerate(data_list):
        axs[1, idx].bar(index + i * bar_width, data, bar_width, label=algorithms[i])
    
    axs[1, idx].set_title('Group Unfairness Reduction' if idx == 0 else 'RMSE Increase')
    axs[1, idx].set_xticks(index + bar_width / 2)
    axs[1, idx].set_xticklabels(categories)
    axs[1, idx].legend()

plt.tight_layout()
plt.show()
