import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import RecSysALS
import RecSysKNN
import RecSysNMF
from RecSysExampleData20Items import RecSysExampleData20Items


class UserFairness():
        
    def __init__(self, n_users, n_movies, top_users, top_movies, l, theta, k):
        self.n_users = n_users
        self.n_movies = n_movies
        self.top_users = top_users
        self.top_movies = top_movies
        self.l = l
        self.theta = theta
        self.k = k
        

#######################################################################################################################
class Polarization():
    
    def evaluate(self, X_est):
        #print("def evaluate(self, X_est):")
        #print("X_est")
        #print(X_est)
        return X_est.var(axis=0,ddof=0).mean()

    def gradient(self, X_est):
        """
        Returns the gradient of the divergence utility defined on the
        estimated ratings of the original users.
        The output is an n by d matrix which is flatten.
        """
        D = X_est - X_est.mean()
        G = D.values
        return  G

#######################################################################################################################
class IndividualLossVariance():
    
    def __init__(self, X, omega, axis):
        self.axis = axis
        self.omega = omega
        self.X = X.mask(~omega)
        self.omega_user = omega.sum(axis=axis)
        
    def get_losses(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        losses = E.mean(axis=self.axis)
        return losses
        
    def evaluate(self, X_est):
        losses = self.get_losses(X_est)
        var =  losses.values.var()
        return var

    def gradient(self, X_est):
        """
        Returns the gradient of the utility.
        The output is an n by d matrix which is flatten.
        """
        X = self.X
        X_est = X_est.mask(~self.omega)
        diff = X_est - X
        if self.axis == 0:
            diff = diff.T
            
        losses = self.get_losses(X_est)
        B = losses - losses.mean()
        C = B.divide(self.omega_user)
        D = diff.multiply(C,axis=0)
        G = D.fillna(0).values
        if self.axis == 0:
            G = G.T
        return  G

#######################################################################################################################
class GroupLossVariance():
    
    def __init__(self, X, omega, G, axis):
        self.X = X 
        self.omega = omega
        self.G = G
        self.axis = axis
        
        if self.axis == 0:
            self.X = self.X.T
            self.omega = self.omega.T
            
        self.group_id ={}
        for group in self.G: #G [user1, user2, user3, user4]
            for user in G[group]:
                self.group_id[user] = group
        
        self.omega_group = {}
        for group in self.G:
            self.omega_group[group] = (~self.X.mask(~self.omega).loc[self.G[group]].isnull()).sum().sum()
        
        omega_user = {}
        for user in self.X.index:
            omega_user[user] = self.omega_group[self.group_id[user]]
        self.omega_user = pd.Series(omega_user)
        
    def get_losses(self, X_est):
        if self.axis == 0:
            X_est = X_est.T
            
        X = self.X.mask(~self.omega)
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        if not E.shape == X.shape:
            print ('dimension error')
            return
        losses = {}
        for group in self.G:
            losses[group] = np.nanmean(E.loc[self.G[group]].values)
        losses = pd.Series(losses)
        return losses
        
    def evaluate(self, X_est):
        losses = self.get_losses(X_est)
        var =  losses.values.var()
        return var

    def gradient(self, X_est):
        """
        Returns the gradient of the utility.
        The output is an n by d matrix which is flatten.
        """
        group_losses = self.get_losses(X_est)
        #n_group = len(self.G)
        
        X = self.X.mask(~self.omega)
        if self.axis == 0:
            X_est = X_est.T
        
        X_est = X_est.mask(~self.omega)
        diff = X_est - X
        if not diff.shape == X.shape:
            print ('dimension error')
            return
        
        user_group_losses ={}
        for user in X.index:
            user_group_losses[user] = group_losses[self.group_id[user]]
        losses = pd.Series(user_group_losses)
        
        B = losses - group_losses.mean()
        C = B.divide(self.omega_user)
        #C = (4.0/n_group) * C
        D = diff.multiply(C,axis=0)
        G = D.fillna(0).values
        if self.axis == 0:
            G = G.T
        return  G

#######################################################################################################################
class ImpartialityAlgorithm():
    
    def __init__(self, X, omega, axis):
        self.axis = axis
        self.omega = omega
        self.X = X.mask(~omega)
        self.omega_user = omega.sum(axis=axis)
        
    def get_differences(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        
        list_dif = []
        
        for row_X, row_Y in zip(X.iterrows(), X_est.iterrows()):
            list_row_dif = []
            for value_X, value_Y in zip(row_X[1], row_Y[1]):
                list_row_dif.append(value_X - value_Y)
            list_dif.append(list_row_dif)
        
        '''for i in list_dif:
            print(i)'''

        list_dif_mean = []
        i = 1
        for list_row_dif in list_dif:
            list_row_dif_valid = [dif for dif in list_row_dif if not(np.isnan(dif))]
            #print(i, " - Média das diferenças: ", np.mean(list_row_dif_valid), list_row_dif_valid)
            list_dif_mean.append(np.mean(list_row_dif_valid))
            i = i + 1

        return list_dif_mean
        
    def evaluate(self, X_est):
        list_dif_mean = self.get_differences(X_est)
        for i in range(0, len(X_est.index)):
            for j in range(0, len(X_est.columns)):
                value = X_est.iloc[i, j] + list_dif_mean[i]
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        return X_est

#######################################################################################################################
class RMSE():
    
    def __init__(self, X, omega):
        self.omega = omega
        self.X = X.mask(~omega)        
        
    def evaluate(self, X_est):
        X_not_na = self.X.values[~np.isnan(self.X.values)]
        X_est_not_na = X_est.values[~np.isnan(self.X.values)]
        
        return np.sqrt(mean_squared_error(X_not_na, X_est_not_na))

#######################################################################################################################