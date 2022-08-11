import numpy as np
import random as random
from sklearn.metrics import mean_squared_error

class AlgorithmImpartiality():
    
    def __init__(self, X, omega, axis):
        self.axis = axis
        self.omega = omega
        self.X = X.mask(~omega)
        self.omega_user = omega.sum(axis=axis)
        
    '''def get_differences(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        
        list_dif = []
        
        for row_X, row_Y in zip(X.iterrows(), X_est.iterrows()):
            list_row_dif = []
            for value_X, value_Y in zip(row_X[1], row_Y[1]):
                list_row_dif.append(value_X - value_Y)
            list_dif.append(list_row_dif)
        
        #for i in list_dif:
        #    print(i)

        list_dif_mean = []
        i = 1
        for list_row_dif in list_dif:
            list_row_dif_valid = [dif for dif in list_row_dif if not(np.isnan(dif))]
            #print(i, " - Média das diferenças: ", np.mean(list_row_dif_valid), list_row_dif_valid)
            list_dif_mean.append(np.mean(list_row_dif_valid))
            i = i + 1

        return list_dif_mean'''
    
    def get_differences_means(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)

        E = (X_est - X)
        losses = E.mean(axis=self.axis)
        return losses

    def get_differences_vars(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        losses = E.mean(axis=self.axis)
        return losses
        
    # Estratégia de perturbação baseada na MÉDIA das diferenças de avaliações
    def get_X_est(self, X_est):
        list_dif_mean = (self.get_differences_means(X_est)).tolist()
        for i in range(0, len(X_est.index)):
            for j in range(0, len(X_est.columns)):
                if (list_dif_mean[i] > 0):
                    value = X_est.iloc[i, j] + random.uniform(0, list_dif_mean[i])
                else:
                    value = X_est.iloc[i, j] + random.uniform(list_dif_mean[i], 0)
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        return X_est

    # Estratégia de perturbação baseada na variância das diferenças de avaliações
    #Variância máxima = 16 (ou seja, 4*4)
    #Variância normalizada: li/4
    def get_X_est2(self, X_est):
        list_dif_mean = (self.get_differences_means(X_est)).tolist()
        list_dif_var = (self.get_differences_vars(X_est)).tolist()
        #print("list_dif_mean: ", list_dif_mean)
        #print("list_dif_var: ", list_dif_var)
        for i in range(0, len(X_est.index)):
            #var_norm = 16.0*list_dif_var[i]/4.0
            var_norm = list_dif_var[i]/4.0
            #print("i: ", i, " - var_norm: ", var_norm)
            for j in range(0, len(X_est.columns)):
                if (list_dif_mean[i] > 0):
                    value = X_est.iloc[i, j] + random.uniform(0, var_norm)
                else:
                    value = X_est.iloc[i, j] + random.uniform(var_norm, 0)
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        return X_est
    
    '''def get_X_est(self, X_est):
        series_dif_mean = self.get_differences(X_est)
        list_dif_mean = series_dif_mean.tolist()

        X_est = X_est + list_dif_mean

        for i in range(0, len(X_est.index)):
            for j in range(0, len(X_est.columns)):
                if (list_dif_mean[i] > 0):
                    value = X_est.iloc[i, j] + random.uniform(0, list_dif_mean[i])
                else:
                    value = X_est.iloc[i, j] + random.uniform(list_dif_mean[i], 0)
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        print(X_est)
        return X_est'''

    def evaluate(self, X_est, n):
        
        list_X_est = []
        print('get_differences end')
        for x in range(0, n):
            print("x: ", x)
            list_X_est.append(self.get_X_est2(X_est.copy()))
        return list_X_est

    '''def evaluate(self, X_est):
        list_dif_mean = self.get_differences(X_est)
        for i in range(0, len(X_est.index)):
            for j in range(0, len(X_est.columns)):
                if (list_dif_mean[i] > 0):
                    value = X_est.iloc[i, j] + random.uniform(0, list_dif_mean[i])
                else:
                    value = X_est.iloc[i, j] + random.uniform(list_dif_mean[i], 0)
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        return X_est'''