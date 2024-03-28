import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt

class RecSysSVD:
    def __init__(self, n_factors=50, ratings=None):
        self.n_factors = n_factors
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        self.ratings = ratings.fillna(0)  # Preenche os valores ausentes com 0 para a SVD
        # Observação: Preencher com a média ou outro valor pode ser considerado, dependendo do caso de uso

    def fit_model(self):
        # Obtem os valores da matriz de classificações
        matrix = self.ratings.values
        
        # Aplica SVD na matriz de classificações
        U, sigma, Vt = svds(matrix, k=self.n_factors)
        
        # Transforma sigma em uma matriz diagonal
        sigma = np.diag(sigma)
        
        # Calcula as classificações aproximadas usando a matriz de fatores
        approx_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # Converte as classificações aproximadas de volta para o formato de DataFrame
        self.predictions = pd.DataFrame(approx_ratings, index=self.ratings.index, columns=self.ratings.columns)
        
        # Calcula o RMSE usando apenas as classificações originais não-nulas (não ausentes)
        mask = self.ratings > 0
        mse = mean_squared_error(self.ratings[mask], self.predictions[mask])
        rmse = sqrt(mse)
        
        return self.predictions, rmse

    def recommend_items(self, user_id, top_n=10):
        # Obtém as predições para um usuário específico
        user_predictions = self.predictions.loc[user_id].sort_values(ascending=False)
        
        # Obtém os itens já classificados pelo usuário
        known_items = self.ratings.loc[user_id] > 0
        
        # Filtra as recomendações para incluir apenas itens não classificados pelo usuário
        recommendations = user_predictions[~known_items].head(top_n)
        
        return recommendations

# Exemplo de uso:
# Supondo que `ratings_df` seja seu DataFrame de classificações, onde as linhas representam usuários
# e as colunas representam itens, com classificações ausentes representadas por NaN
# ratings_df = pd.DataFrame(...)
# rec_sys_svd = RecSysSVD(n_factors=50, ratings=ratings_df)
# predictions, rmse = rec_sys_svd.fit_model()
# print(f"RMSE: {rmse}")
# recommendations = rec_sys_svd.recommend_items(user_id='algum_id_de_usuario', top_n=10)
# print(recommendations)
