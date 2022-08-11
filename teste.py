import pandas as pd
df = pd.DataFrame({'angles': [0, 3, 4],
                   'degrees': [360, 180, 360]},
                   index=['circle', 'triangle', 'rectangle'])

print(df)

df = df - [1, 2]
print(df)