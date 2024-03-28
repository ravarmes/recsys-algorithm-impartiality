import pandas as pd

# Ler o arquivo 'ratings.csv' (ajuste o caminho conforme necessário)
ratings_df = pd.read_csv('ratings.csv', sep=';')

# Eliminar duplicatas mantendo a primeira ocorrência
ratings_df_cleaned = ratings_df.drop_duplicates(subset=['UserID', 'BookID'], keep='first')

# Salvar o DataFrame limpo de volta em um arquivo, se necessário
ratings_df_cleaned.to_csv('ratings_cleaned.csv', sep=';', index=False)
