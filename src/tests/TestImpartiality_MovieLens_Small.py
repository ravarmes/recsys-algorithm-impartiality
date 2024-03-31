from RecSys import RecSys
from AlgorithmUserFairness import Polarization, IndividualLossVariance, GroupLossVariance, RMSE
from AlgorithmImpartiality import AlgorithmImpartiality
import numpy as np
import pandas as pd
import gurobipy as gp

#--------------------------GUROBI-----------------------------------------------------

def losses_to_Z(list_losses):
    Z = []
    linha = []
    for i in range (0, n_users):
        for losses in list_losses:
            linha.append(losses.values[i])
        Z.append(linha.copy())
        linha.clear()
    return Z

def matrices_Zs(Z, G): # retorar uma matriz Z para cada grupo
    list_Zs = []
    for group in G: # G = {1: [1,2], 2: [3,4,5]}
        Z_ = []
        list_users = G[group]
        for user in list_users:
            Z_.append(Z[user-1].copy())    
        list_Zs.append(Z_)
    return list_Zs


#--------------------------GUROBI-----------------------------------------------------

# reading data from a base with 20 movies and 40 users
Data_path = 'Data/MovieLens-Small'
n_users=  40
n_items= 20
top_users = False # True: to use users with more ratings; False: otherwise
top_items = False # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysExampleAntidoteData20Items' # this algorithm should only be used for a database with 40 users and 20 items 'Data/Movie20Items'
#algorithm = 'RecSysALS'


# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_movielens_small(n_users, n_items, top_users, top_items, Data_path) # returns matrix of ratings with n_users rows and n_moveis columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [before the impartiality algorithm] ------------")

# To capture polarization, we seek to measure the extent to which the user ratings disagree
polarization = Polarization()
Rpol = polarization.evaluate(X_est)
print("Polarization (Rpol):", Rpol)

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# advantaged group: 5% users with the highest number of item ratings
# disadvantaged group: 95% users with the lowest number of item ratings
G = {1: [1,2], 2: [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
RgrpNR = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NR - 95-5%):", RgrpNR)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)
print("RMSE: ", result)


##############################################################################################################################
algorithmImpartiality = AlgorithmImpartiality(X, omega, 1)
list_X_est = algorithmImpartiality.get_X_est(X_est) # calculates 1 estimated matrices

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [after the impartiality algorithm] -------------")

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# advantaged group: 5% users with the highest number of item ratings
# disadvantaged group: 95% users with the lowest number of item ratings
G = {1: [1,2], 2: [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)

RgrpNR = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NR):", RgrpNR)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)
print("RMSE: ", result)