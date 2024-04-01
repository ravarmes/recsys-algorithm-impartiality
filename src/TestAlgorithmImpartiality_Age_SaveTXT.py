from RecSys import RecSys
from AlgorithmUserFairness import Polarization, IndividualLossVariance, GroupLossVariance, RMSE
from AlgorithmImpartiality import AlgorithmImpartiality

# user and item filtering
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use MovieLens with more ratings; False: otherwise

# estimated number of matrices (h)
# h = 3
# h = 5
# h = 10
h = 15
# h = 20

# dataset
dataset = 'MovieLens-1M'    # reading data from 3883 movies and 6040 users
# dataset = 'Goodbooks-10k' # reading data from 10000 GoodBooks and 53424 users
# dataset = 'Songs'         # reading data from 19993 songs and 16000 users

# recommendation algorithm
# algorithm = 'RecSysALS' # Alternating Least Squares (ALS) for Collaborative Filtering
# algorithm = 'RecSysKNN' # K-Nearest Neighbors for Recommender Systems
# algorithm = 'RecSysNMF' # Non-Negative Matrix Factorization for Recommender Systems
# algorithm = 'RecSysSGD' # Stochastic Gradient Descent for Recommender Systems
# algorithm = 'RecSysSVD' # Singular Value Decomposition for Recommender Systems
# algorithm = 'RecSysNCF' # Neural Collaborative Filtering
# algorithms = ['RecSysALS', 'RecSysKNN', 'RecSysNMF', 'RecSysSGD', 'RecSysNCF']
# algorithms = ['RecSysALS', 'RecSysKNN', 'RecSysNMF', 'RecSysNCF']
algorithms = ['RecSysKNN']

file = open(f'results-{h}.txt', 'w', encoding='utf-8')

for algorithm in algorithms:

    print(f'\n\n------------------------------')
    print(f'Algorithm: {algorithm}')
    file.write(f'\n\n------------------------------\n')
    file.write(f'Algorithm: {algorithm}\n')

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

    formatted_Rpol = format(Rpol, ".9f").replace('.', ',')
    file.write(f'Polarization (Rpol): {formatted_Rpol}\n')


    # Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
    ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
    Rindv = ilv.evaluate(X_est)
    print(f'Individual Loss Variance (Rindv): {Rindv:.9f}')

    formatted_Rindv = format(Rindv, ".9f").replace('.', ',')
    file.write(f'Individual Loss Variance (Rindv): {formatted_Rindv}\n')

    losses = ilv.get_losses(X_est)

    # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
    # The loss of group i as the mean squared estimation error over all known ratings in group i
    # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # G = {1: advantaged_group, 2: disadvantaged_group}
    list_users = X_est.index.tolist()
    age_00_17 = users_info[users_info['Age'] ==  1].index.intersection(list_users).tolist()
    age_18_24 = users_info[users_info['Age'] == 18].index.intersection(list_users).tolist()
    age_25_34 = users_info[users_info['Age'] == 25].index.intersection(list_users).tolist()
    age_35_44 = users_info[users_info['Age'] == 35].index.intersection(list_users).tolist()
    age_45_49 = users_info[users_info['Age'] == 45].index.intersection(list_users).tolist()
    age_50_55 = users_info[users_info['Age'] == 50].index.intersection(list_users).tolist()
    age_56_00 = users_info[users_info['Age'] == 56].index.intersection(list_users).tolist()

    G = {1: age_00_17, 2: age_18_24, 3: age_25_34, 4: age_35_44, 5: age_45_49, 6: age_50_55, 7: age_56_00}
    G_index = {1: [14, 194, 273, 132, 262], 2: [251, 175, 94, 86, 270, 48, 92, 96, 129, 107, 134, 64, 124, 288, 159, 26, 101, 61, 76, 126, 207, 191, 174, 168, 70, 215, 209, 171, 82, 149, 71, 33, 216, 157, 189, 23, 50, 231, 222, 201, 140, 163, 246, 244, 282, 265, 8, 90, 290, 158, 255, 237, 275], 3: [183, 200, 220, 212, 141, 230, 7, 60, 120, 170, 205, 16, 41, 145, 37, 34, 261, 69, 176, 32, 73, 35, 213, 133, 279, 285, 74, 276, 135, 252, 80, 225, 81, 63, 56, 138, 102, 295, 3, 296, 155, 161, 108, 51, 97, 269, 195, 283, 236, 21, 289, 232, 198, 66, 249, 238, 72, 293, 203, 188, 258, 10, 196, 139, 116, 291, 226, 233, 31, 142, 186, 118, 217, 151, 181, 45, 254, 42, 190, 89, 55, 147, 294, 169, 103, 93, 85, 28, 286, 39, 241, 267, 211, 223, 143, 229, 104, 110, 277, 136, 206, 40, 65, 11, 264, 29, 15, 22, 24, 234, 287, 130, 280, 131, 187, 164, 128, 43, 44, 298, 219, 75, 179, 109, 122, 173, 260, 202, 53, 79, 59, 253, 192, 204, 9, 119, 6, 111, 268, 240, 193, 106], 4: [272, 99, 299, 87, 199, 13, 84, 271, 1, 227, 18, 248, 117, 17, 127, 77, 177, 292, 156, 27, 214, 4, 5, 91, 297, 105, 49, 121, 98, 210, 52, 166, 95, 112, 88, 68, 208, 243, 152, 245, 228, 100, 250, 2, 278, 36, 144, 146, 256, 38, 57, 78, 172, 182, 150, 153, 25, 263], 5: [137, 221, 30, 160, 19, 67, 165, 113, 54, 281, 184, 167, 197, 125, 58, 47, 242, 83, 20, 235, 239, 148, 46, 62], 6: [247, 180, 266, 123, 257, 185, 114, 224, 178, 0, 115, 274], 7: [162, 12, 259, 154, 218, 284]}
    
    glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpNR = glv.evaluate(X_est)

    rmse = RMSE(X, omega)
    result = rmse.evaluate(X_est)

    print(f'Group Loss Variance (Rgrp): {RgrpNR:.9f}')
    print(f'RMSE: {result:.9f}\n')

    formatted_RgrpNR = format(RgrpNR, ".9f").replace('.', ',')
    file.write(f'Group Loss Variance (Rgrp): {formatted_RgrpNR}\n')

    formatted_result = format(result, ".9f").replace('.', ',')
    file.write(f'RMSE: {formatted_result}\n')

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
    X_gurobi = AlgorithmImpartiality.make_matrix_X_gurobi_seven_groups(list_X_est, G, list_Zs)

    # Show group injustice and optimized recommendation matrix RMSE
    RgrpNR = glv.evaluate(X_gurobi)

    rmse = RMSE(X, omega)
    result = rmse.evaluate(X_gurobi)

    print(f'Group Loss Variance (Rgrp): {RgrpNR:.9f}')
    print(f'RMSE: {result:.9f}\n')

    file.write(f'Gurobi\n')

    formatted_RgrpNR = format(RgrpNR, ".9f").replace('.', ',')
    file.write(f'Group Loss Variance (Rgrp): {formatted_RgrpNR}\n')

    formatted_result = format(result, ".9f").replace('.', ',')
    file.write(f'RMSE: {formatted_result}\n')

    # X_gurobi.to_excel(f'Xest_{dataset}_Gurobi.xlsx', index=True)


file.close()


