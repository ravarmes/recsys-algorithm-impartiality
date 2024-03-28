import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

class RecSysSGD:
    def __init__(self, n_factors=10, learning_rate=0.01, n_epochs=20, lambda_=0.1, ratings=None):
        self.n_factors = n_factors  # Número de fatores latentes
        self.learning_rate = learning_rate  # Taxa de aprendizado
        self.n_epochs = n_epochs  # Número de épocas
        self.lambda_ = lambda_  # Termo de regularização
        self.scaler = MinMaxScaler(feature_range=(1, 5))  # Escalonador para normalizar as classificações
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        self.ratings = ratings.fillna(0)  # Substitui NaN por 0
        # Normaliza as classificações para o intervalo [1, 5]
        self.ratings_scaled = pd.DataFrame(self.scaler.fit_transform(self.ratings), index=ratings.index, columns=ratings.columns)

    def fit_model(self):
        m, n = self.ratings_scaled.shape
        # Inicializa os fatores latentes de usuários e itens
        U = np.random.rand(m, self.n_factors)
        V = np.random.rand(n, self.n_factors)
        
        for epoch in range(self.n_epochs):
            for i in range(m):
                for j in range(n):
                    if self.ratings_scaled.iloc[i, j] > 0:  # Considera apenas classificações conhecidas
                        # Calcula o erro
                        eij = self.ratings_scaled.iloc[i, j] - np.dot(U[i, :], V[j, :])
                        # Atualiza os fatores latentes
                        U[i, :] += self.learning_rate * (eij * V[j, :] - self.lambda_ * U[i, :])
                        V[j, :] += self.learning_rate * (eij * U[i, :] - self.lambda_ * V[j, :])
        
        # Calcula a matriz de predições
        pred_ratings = np.dot(U, V.T)
        # Ajusta as predições para o intervalo original de classificações
        pred_ratings_adjusted = self.scaler.inverse_transform(pred_ratings)
        self.pred = pd.DataFrame(pred_ratings_adjusted, index=self.ratings.index, columns=self.ratings.columns)
        
        return self.pred

# Exemplo de uso:
# ratings_df = pd.DataFrame(...)  # Substitua por sua matriz de classificações
# recSys = RecSysSGDSimplified(n_factors=10, ratings=ratings_df)
# predictions = recSys.fit_model()
