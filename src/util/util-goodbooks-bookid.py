import pandas as pd

# Ler o arquivo 'ratings.csv' do diretório atual
ratings_df = pd.read_csv('ratings.csv', sep=';')

# Extrair os user_id distintos e ordená-los
unique_book_ids = ratings_df['BookID'].unique()
unique_book_ids.sort()

# Salvar os user_id em um novo DataFrame
books_df = pd.DataFrame(unique_book_ids, columns=['BookID'])

# Salvar os dados em um arquivo 'books.csv' no diretório atual
books_df.to_csv('books.csv', index=False)
