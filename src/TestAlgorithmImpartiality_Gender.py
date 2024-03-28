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

# G group: identifying the groups (Age: users grouped by age)
#print(users_info)

list_users = X_est.index.tolist()

masculine = users_info[users_info['Gender'] == 1].index.intersection(list_users).tolist()
feminine = users_info[users_info['Gender'] == 2].index.intersection(list_users).tolist()

G = {1: masculine, 2: feminine}
G_index = {1: [183, 200, 272, 247, 220, 212, 251, 141, 175, 137, 99, 299, 94, 230, 87, 86, 7, 270, 60, 120, 48, 170, 205, 16, 30, 160, 41, 37, 96, 261, 69, 129, 176, 107, 19, 266, 123, 32, 13, 84, 271, 134, 64, 124, 288, 1, 227, 257, 279, 159, 285, 74, 276, 18, 248, 135, 67, 252, 80, 117, 194, 26, 17, 127, 56, 61, 138, 102, 295, 76, 126, 3, 77, 296, 207, 177, 292, 156, 161, 108, 27, 165, 191, 51, 4, 174, 5, 269, 168, 283, 70, 21, 209, 162, 289, 232, 198, 82, 66, 249, 273, 54, 238, 281, 72, 297, 184, 293, 105, 149, 49, 203, 188, 258, 10, 196, 210, 71, 139, 291, 226, 33, 233, 52, 142, 186, 118, 166, 189, 23, 95, 112, 50, 151, 181, 45, 88, 12, 89, 231, 55, 68, 147, 294, 169, 125, 208, 103, 93, 85, 28, 286, 259, 39, 58, 222, 267, 211, 223, 201, 152, 245, 224, 178, 143, 140, 229, 47, 154, 110, 277, 136, 0, 206, 40, 100, 250, 2, 65, 11, 278, 29, 246, 36, 83, 15, 22, 20, 24, 234, 287, 146, 256, 130, 280, 38, 131, 187, 218, 57, 164, 128, 78, 282, 172, 43, 44, 182, 132, 115, 298, 219, 75, 109, 265, 8, 90, 260, 239, 148, 202, 153, 262, 53, 79, 59, 263, 255, 284, 204, 9, 237, 6, 111, 268, 240, 274, 46, 275, 62], 2: [180, 221, 92, 145, 199, 34, 73, 14, 35, 213, 133, 225, 81, 63, 101, 155, 113, 214, 97, 195, 236, 215, 171, 91, 121, 185, 98, 116, 31, 114, 216, 157, 217, 167, 254, 42, 190, 197, 241, 243, 228, 104, 242, 264, 163, 144, 244, 235, 179, 122, 173, 150, 290, 158, 25, 253, 192, 119, 193, 106]}
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





