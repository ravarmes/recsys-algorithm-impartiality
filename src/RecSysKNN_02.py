import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod

class RecSysKNN:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, user_based=True):
        self.k = k
        self.user_based = user_based
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~self.ratings.isnull()).sum(axis=0)

    def get_similarity_matrix(self):
        if self.user_based:
            similarity = pd.DataFrame(cosine_similarity(self.ratings.fillna(0)), index=self.ratings.index, columns=self.ratings.index)
        else:
            similarity = pd.DataFrame(cosine_similarity(self.ratings.fillna(0).T), index=self.ratings.columns, columns=self.ratings.columns)
        return similarity
    
    def knn_filtering(self, similarity):
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k+1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        similarity = similarity.div(similarity.sum(axis=1), axis=0)
        return similarity
    
    # def fit_model(self, max_iter=50, threshold=1e-5):
    #     similarity = self.get_similarity_matrix()
    #     knn_similarity = self.knn_filtering(similarity)
        
    #     if self.user_based:
    #         pred = knn_similarity.dot(self.ratings.fillna(0)).div(knn_similarity.sum(axis=1), axis=0)
    #     else:
    #         pred = self.ratings.fillna(0).dot(knn_similarity).div(knn_similarity.sum(axis=0), axis=1)
        
    #     imputer = SimpleImputer(strategy='mean')
    #     self.pred = pd.DataFrame(imputer.fit_transform(pred), index=self.ratings.index, columns=self.ratings.columns)
        
    #     self.U = knn_similarity if self.user_based else self.ratings
    #     self.V = self.ratings.T if self.user_based else knn_similarity.T
        
    #     return self.pred

    def fit_model(self, max_iter=50, threshold=1e-5):
        # Calcula a matriz de similaridade
        similarity = self.get_similarity_matrix()
        # Aplica a filtragem KNN para obter as similaridades ajustadas
        knn_similarity = self.knn_filtering(similarity)
        
        # Inicializa a matriz de predições com zeros
        pred = pd.DataFrame(np.zeros(self.ratings.shape), index=self.ratings.index, columns=self.ratings.columns)
        
        # Calcula as predições
        if self.user_based:
            # Para cada usuário
            for user in self.ratings.index:
                user_ratings = self.ratings.loc[user].dropna()
                if not user_ratings.empty:
                    # Considera apenas os itens classificados pelo usuário
                    sim_users = knn_similarity.loc[user]
                    for item in user_ratings.index:
                        # Calcula a média ponderada das classificações dos usuários mais similares que também classificaram o item
                        sim_users_item = self.ratings[item].notnull() * sim_users
                        if sim_users_item.sum() > 0:
                            pred.loc[user, item] = (sim_users_item * self.ratings[item]).sum() / sim_users_item.sum()
        else:
            # Item-based: uma lógica similar pode ser aplicada para item-based
            pass
        
        # Imputação: Substitui zeros pela média geral (você pode ajustar para média por usuário ou item)
        pred = pred.replace(0, np.nan)
        if self.user_based:
            mean_ratings = self.ratings.mean(axis=1)
            for user in pred.index:
                pred.loc[user].fillna(mean_ratings[user], inplace=True)
        else:
            # Para item-based, considere usar a média por item
            pass
        
        self.pred = pred
        return self.pred


# Exemplo de uso
# ratings_df = pd.DataFrame(...)  # Substitua por sua matriz de classificações
# rec_sys_knn = RecSysKNN(k=5, ratings=ratings_df, user_based=True)
# predictions = rec_sys_knn.fit_model()
