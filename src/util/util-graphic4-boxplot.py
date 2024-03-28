import matplotlib.pyplot as plt

Rgrp_MovieLens_Hierarchical = [-17.32, -19.00, -20.07, -22.90, -23.07]
Rgrp_Songs_Hierarchical = [-22.06, -30.20, -37.92, -41.64, -43.68]
Rgrp_MovieLens_95_5 = [-70.93, -71.80, -72.11, -72.59, -72.05]
Rgrp_Songs_95_5 = [-84.80, -93.86, -93.87, -93.49, -93.11]

RMSE_MovieLens_Hierarchical = [0.42, 0.35, 0.30, 0.22, 0.19]
RMSE_Songs_Hierarchical = [-3.99, -4.27, -4.73, -4.88, -5.02]
RMSE_MovieLens_95_5 = [0.49, 0.47, 0.49, 0.54, 0.51]
RMSE_Songs_95_5 = [0.13, 0.08, 1.18, 0.81, 1.16]

# Preparing the data for the boxplots
data_Rgrp = [Rgrp_MovieLens_Hierarchical, Rgrp_Songs_Hierarchical, Rgrp_MovieLens_95_5, Rgrp_Songs_95_5]
data_RMSE = [RMSE_MovieLens_Hierarchical, RMSE_Songs_Hierarchical, RMSE_MovieLens_95_5, RMSE_Songs_95_5]

fig, axs = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
fig.suptitle('Distribution of Rgrp and RMSE by Grouping Configuration and Dataset', fontsize=16)

# Boxplot for Rgrp
axs[0].boxplot(data_Rgrp, patch_artist=True)
axs[0].set_title('Rgrp Distribution', fontsize=14)
axs[0].set_xticklabels(['MovieLens Hier.', 'Songs Hier.', 'MovieLens 95-5', 'Songs 95-5'], fontsize=12)
axs[0].set_ylabel('Percentage Change (%)', fontsize=12)

# Boxplot for RMSE
axs[1].boxplot(data_RMSE, patch_artist=True)
axs[1].set_title('RMSE Distribution', fontsize=14)
axs[1].set_xticklabels(['MovieLens Hier.', 'Songs Hier.', 'MovieLens 95-5', 'Songs 95-5'], fontsize=12)
axs[1].set_ylabel('Percentage Change (%)', fontsize=12)

plt.show()
