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
        self.ratings = None
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.original_ratings = ratings
        # Normalização das classificações para remover viés de usuário/item
        if self.user_based:
            self.ratings = ratings.apply(lambda x: (x - x.mean()) / (x.std()), axis=1).fillna(0)
        else:
            self.ratings = ratings.apply(lambda x: (x - x.mean()) / (x.std()), axis=0).fillna(0)
        
        self.similarity_matrix = None

    def get_similarity_matrix(self):
        if self.similarity_matrix is None:
            # Calcula a similaridade cosseno
            if self.user_based:
                sim_matrix = cosine_similarity(self.ratings)
                self.similarity_matrix = pd.DataFrame(sim_matrix, index=self.original_ratings.index, columns=self.original_ratings.index)
            else:
                sim_matrix = cosine_similarity(self.ratings.T)
                self.similarity_matrix = pd.DataFrame(sim_matrix, index=self.original_ratings.columns, columns=self.original_ratings.columns)
        return self.similarity_matrix
    
    def knn_filtering(self):
        similarity = self.get_similarity_matrix().copy()
        for idx, row in similarity.iterrows():
            nearest = row.nlargest(self.k + 1).index
            similarity.loc[idx, ~similarity.columns.isin(nearest)] = 0
        return similarity
    
    def fit_model(self):
        knn_similarity = self.knn_filtering()
        
        if self.user_based:
            pred_ratings = knn_similarity.dot(self.original_ratings.fillna(0))
            normalization_factor = knn_similarity.sum(axis=1)
            pred_ratings = pred_ratings.div(normalization_factor, axis=0)
        else:
            pred_ratings = self.original_ratings.fillna(0).dot(knn_similarity)
            normalization_factor = knn_similarity.sum(axis=0)
            pred_ratings = pred_ratings.div(normalization_factor, axis=1)
        
        self.predictions = pred_ratings
        return self.predictions

# Exemplo de uso
# ratings_df = pd.DataFrame(...)  # Substitua por sua matriz de classificações
# advanced_knn = AdvancedKNN(k=5, ratings=ratings_df, user_based=True)
# predictions = advanced_knn.fit_model()
