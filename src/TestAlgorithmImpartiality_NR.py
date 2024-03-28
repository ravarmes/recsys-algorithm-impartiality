from RecSys import RecSys
from AlgorithmUserFairness import Polarization, IndividualLossVariance, GroupLossVariance, RMSE
from AlgorithmImpartiality import AlgorithmImpartiality

# user and item filtering
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use MovieLens with more ratings; False: otherwise

# dataset
dataset = 'MovieLens-1M'    # reading data from 3883 movies and 6040 users
# dataset = 'Goodbooks-10k' # reading data from 10000 GoodBooks and 53424 users
# dataset = 'Songs'         # reading data from 19993 songs and 16000 users

# recommendation algorithm
algorithm = 'RecSysALS' # Alternating Least Squares (ALS) for Collaborative Filtering
# algorithm = 'RecSysKNN' # K-Nearest Neighbors for Recommender Systems
# algorithm = 'RecSysNMF' # Non-Negative Matrix Factorization for Recommender Systems
# algorithm = 'RecSysSGD' # Stochastic Gradient Descent for Recommender Systems
# algorithm = 'RecSysSVD' # Singular Value Decomposition for Recommender Systems
# algorithm = 'RecSysNCF' # Neural Collaborative Filtering

# estimated number of matrices (h)
h = 3
# h = 5
# h = 10
# h = 15
# h = 20

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

Data_path = "Data/"+ dataset    

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

# X.to_excel(f'X_{dataset}.xlsx', index=True)
# X_est.to_excel(f'Xest_{dataset}.xlsx', index=True)

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [before the impartiality algorithm] ------------")

# To capture polarization, we seek to measure the extent to which the user ratings disagree
polarization = Polarization()
Rpol = polarization.evaluate(X_est)
print(f'Polarization (Rpol): {Rpol:.9f}')

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print(f'Individual Loss Variance (Rindv): {Rindv:.9f}')

losses = ilv.get_losses(X_est)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# G = {1: advantaged_group, 2: disadvantaged_group}
list_users = X_est.index.tolist()
advantaged_group = list_users[0:15]
disadvantaged_group = list_users[15:300]
G = {1: advantaged_group, 2: disadvantaged_group}

G_index = {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
RgrpNR = glv.evaluate(X_est)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)

print(f'Group Loss Variance (Rgrp): {RgrpNR:.9f}')
print(f'RMSE: {result:.9f}\n')

##############################################################################################################################
algorithmImpartiality = AlgorithmImpartiality(X, omega, 1)
list_X_est = algorithmImpartiality.evaluate(X_est, h) # calculates a list of h estimated matrices

#list_X_est[0].to_excel('X1_MovieLens.xlsx', index=True)

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [after the impartiality algorithm] -------------")

# for X_est in list_X_est:
#     RgrpNR = glv.evaluate(X_est)
#     print("Group Loss Variance (Rgrp):", RgrpNR, end='; ') # NR: cluster by number ratings
#     rmse = RMSE(X, omega)
#     result = rmse.evaluate(X_est)
#     print("RMSE: ", result)

# FRAMEWORK: preparing data for Rgrp minimization

print("\n--------------------------------- optimization result ----------------------------------")

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
#ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
list_losses = []
for X_est in list_X_est:
    losses = ilv.get_losses(X_est)
    list_losses.append(losses)

Z = AlgorithmImpartiality.losses_to_Z(list_losses, n_users)

list_Zs = AlgorithmImpartiality.matrices_Zs(Z, G_index)

# Calculate the recommendation matrix optimized by gurobi
X_gurobi = AlgorithmImpartiality.make_matrix_X_gurobi(list_X_est, G, list_Zs)

# Show group injustice and optimized recommendation matrix RMSE
RgrpNR = glv.evaluate(X_gurobi)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_gurobi)

print(f'Group Loss Variance (Rgrp): {RgrpNR:.9f}')
print(f'RMSE: {result:.9f}\n')

# X_gurobi.to_excel(f'Xest_{dataset}_Gurobi.xlsx', index=True)





