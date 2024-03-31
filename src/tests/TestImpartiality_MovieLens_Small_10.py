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
list_X_est = algorithmImpartiality.evaluate(X_est) # calculates a list of 10 estimated matrices


print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [after the impartiality algorithm] -------------")

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# advantaged group: 5% users with the highest number of item ratings
# disadvantaged group: 95% users with the lowest number of item ratings
G = {1: [1,2], 2: [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)

for X_est in list_X_est:
    RgrpNR = glv.evaluate(X_est)
    print("Group Loss Variance (Rgrp NR - 95-5%):", RgrpNR, end='; ')
    rmse = RMSE(X, omega)
    result = rmse.evaluate(X_est)
    print("RMSE: ", result)


# FRAMEWORK: preparing data for Rgrp minimization
print("\n\n--------------------------------- optimization result ----------------------------------")

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
list_losses = []
for X_est in list_X_est:
    losses = ilv.get_losses(X_est)
    list_losses.append(losses)

Z = losses_to_Z(list_losses)

list_Zs = matrices_Zs(Z, G)

# Rótulos dos usuários e Rindv (injustiças individuais)
users_g1 = ["U{}".format(i + 1) for i in range(2)]
ls_g1 = ["l{}".format(j + 1) for j in range(len(list_X_est))]
users_g2 = ["U{}".format(i + 1) for i in range(38)]
ls_g2 = ["l{}".format(j + 1) for j in range(len(list_X_est))]

# Dicionário com as perdas individuais
Z1 = list_Zs[0]
preferencias1 = dict()
for i, user in enumerate(users_g1):
    for j, l in enumerate(ls_g1):
        preferencias1[user, l] = Z1[i][j]

Z2 = list_Zs[1]
preferencias2 = dict()
for i, user in enumerate(users_g2):
    for j, l in enumerate(ls_g2):
        preferencias2[user, l] = Z2[i][j]

# Inicializa o modelo
m = gp.Model()

# Variáveis de decisão
x1 = m.addVars(users_g1, ls_g1, vtype=gp.GRB.BINARY)
x2 = m.addVars(users_g2, ls_g2, vtype=gp.GRB.BINARY)


# Função objetivo
# Neste caso a função objetivo busca minimizar a variância entre aS injustiçaS doS grupoS (Li) 
# Li também pode ser entendido como a média das injustiças individuais do grupo i. 
# Rgrp: a variância de todas as injustiças dos grupos (Li).

L1 = x1.prod(preferencias1)/len(users_g1)
L2 = x2.prod(preferencias2)/len(users_g2)
LMean = (L1 + L2) / 2
Rgrp = ((L1 - LMean)**2 + (L2 - LMean)**2)/2

m.setObjective( Rgrp , sense=gp.GRB.MINIMIZE)

# Restrições que garantem que todos os usuários terão uma Rindv (cálculo de injustiça individual)
c1 = m.addConstrs(
    x1.sum(i, '*') == 1 for i in users_g1)

c2 = m.addConstrs(
    x2.sum(i, '*') == 1 for i in users_g2)

# Executa o modelo
m.optimize()

print("Grupo 01")
for i in users_g1:
    for j in ls_g1:
        print("{};".format(round(x1[i, j].X)), end="")
    print("")

print("Grupo 02")
for i in users_g2:
    for j in ls_g2:
        print("{};".format(round(x2[i, j].X)), end="")
    print("")

# Imprime a injustiça do grupo

print(f'Rgrp NR: {round(m.objVal):.2f}')

#X_est.to_excel("output.xlsx")





