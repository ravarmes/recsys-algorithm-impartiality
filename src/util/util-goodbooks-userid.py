import pandas as pd

# Ler o arquivo 'ratings.csv' do diretório atual
ratings_df = pd.read_csv('ratings.csv', sep=';')

# Extrair os user_id distintos e ordená-los
unique_user_ids = ratings_df['UserID'].unique()
unique_user_ids.sort()

# Salvar os user_id em um novo DataFrame
users_df = pd.DataFrame(unique_user_ids, columns=['UserID'])

# Salvar os dados em um arquivo 'users.csv' no diretório atual
users_df.to_csv('users.csv', index=False)
