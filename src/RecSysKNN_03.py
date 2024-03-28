import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import normalize  # Adicione esta linha


class RecSysKNN:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, user_based=True):
        self.k = k
        self.user_based = user_based
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.ratings = ratings.fillna(ratings.mean())  # Use a média para preencher valores ausentes
        if not self.user_based:
            self.ratings = self.ratings.T  # Transpose for item-based approach
        
        self.similarity_matrix = None

    def get_similarity_matrix(self):
        if self.similarity_matrix is None:
            # Calcula a similaridade cosseno e normaliza a matriz de similaridade
            sim_matrix = cosine_similarity(self.ratings)
            self.similarity_matrix = pd.DataFrame(normalize(sim_matrix, norm='l1'), index=self.ratings.index, columns=self.ratings.index)
        return self.similarity_matrix
    
    def knn_filtering(self):
        similarity = self.get_similarity_matrix().copy()
        for idx, row in similarity.iterrows():
            # Mantém os k vizinhos mais próximos e zera os outros
            nearest = row.nlargest(self.k + 1).index
            similarity.loc[idx, ~similarity.columns.isin(nearest)] = 0
        return similarity
    
    def fit_model(self):
        knn_similarity = self.knn_filtering()
        
        # Preenche os ratings ausentes na matriz original
        filled_ratings = self.ratings.fillna(self.ratings.mean())
        
        # Calcula as previsões
        if self.user_based:
            predictions = knn_similarity.dot(filled_ratings).div(knn_similarity.sum(axis=1), axis=0)
        else:
            predictions = filled_ratings.dot(knn_similarity).div(knn_similarity.sum(axis=0), axis=1)
            
        self.predictions = predictions
        return self.predictions

# Exemplo de uso
# ratings_df = pd.DataFrame(...)  # Substitua por sua matriz de classificações
# rec_sys_knn = RecSysKNN(k=5, ratings=ratings_df, user_based=True)
# predictions = rec_sys_knn.fit_model()
