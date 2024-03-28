import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Algorithm': ['ALS', 'ALS', 'ALS', 'ALS', 'ALS', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN',
                  'ALS', 'ALS', 'ALS', 'ALS', 'ALS', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN',
                  'ALS', 'ALS', 'ALS', 'ALS', 'ALS', 'KNN', 'KNN', 'KNN', 'KNN', 'KNN'] * 2,
    'Dataset': ['MovieLens']*10 + ['Songs']*10 + ['GoodBooks']*10 + ['MovieLens']*10 + ['Songs']*10 + ['GoodBooks']*10,  # Duplicado corretamente
    'h': [3, 5, 10, 15, 20]*6 + [3, 5, 10, 15, 20]*6,  # Duplicado corretamente
    'Metric': ['R_{grp}']*30 + ['RMSE']*30,
    'Value': [
        # R_{grp} values for ALS and KNN for MovieLens
        0.000008842, 0.000008578, 0.000008483, 0.000008338, 0.000008502,  # ALS
        0.000001200, 0.000000616, 0.000000736, 0.000000624, 0.000000555,  # KNN
        # R_{grp} values for ALS and KNN for Songs
        0.000276807, 0.000058313, 0.000050649, 0.000062162, 0.000069824,  # ALS
        0.000000489, 0.000000418, 0.000000426, 0.000000452, 0.000000476,  # KNN
        # R_{grp} values for ALS and KNN for GoodBooks
        0.000224587, 0.000146169, 0.000102859, 0.000061734, 0.000061405,  # ALS
        0.000081283, 0.000077750, 0.000074121, 0.000071814, 0.000072164,  # KNN
        # RMSE values for ALS and KNN for MovieLens
        0.890821795, 0.890640562, 0.890861685, 0.891311053, 0.891029760,  # ALS
        0.762339873, 0.762144990, 0.746680594, 0.762123255, 0.758920063,  # KNN
        # RMSE values for ALS and KNN for Songs
        0.741968918, 0.746204973, 0.748920480, 0.749877643, 0.752209714,  # ALS
        0.274435523, 0.274535220, 0.274106664, 0.274573152, 0.275721166,  # KNN
        # RMSE values for ALS and KNN for GoodBooks
        0.815134287, 0.818954503, 0.824180214, 0.825082121, 0.821748151,  # ALS
        0.356572213, 0.353435120, 0.357077538, 0.357172978, 0.360204002,  # KNN
    ]
}


# Convertendo o dicionário em DataFrame para facilitar a manipulação
df = pd.DataFrame(data)

# Criando gráficos de violino para R_{grp} e RMSE separadamente
fig, axs = plt.subplots(2, 1, figsize=(14, 12))

# Filtrando os dados para R_{grp} e plotando
df_rgrp = df[df['Metric'] == 'R_{grp}']
sns.violinplot(x='Dataset', y='Value', hue='Algorithm', data=df_rgrp, ax=axs[0], split=True)
axs[0].set_title('Distribution of $R_{grp}(\mu)$ by Algorithm and Dataset')
axs[0].set_ylabel('$R_{grp}(\mu)$')

# Filtrando os dados para RMSE e plotando
df_rmse = df[df['Metric'] == 'RMSE']
sns.violinplot(x='Dataset', y='Value', hue='Algorithm', data=df_rmse, ax=axs[1], split=True)
axs[1].set_title('Distribution of $RMSE(\mu)$ by Algorithm and Dataset')
axs[1].set_ylabel('$RMSE(\mu)$')

plt.tight_layout()
plt.show()
