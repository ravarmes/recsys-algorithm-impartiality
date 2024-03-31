from RecSys import RecSys
from AlgorithmUserFairness import Polarization, IndividualLossVariance, GroupLossVariance, RMSE
from AlgorithmImpartiality import AlgorithmImpartiality
import gurobipy as gp
import pandas as pd


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

def matrices_Zs(Z, G): # return a Z matrix for each group
    list_Zs = []
    for group in G: # G = {1: [1,2], 2: [3,4,5]}
        Z_ = []
        list_users = G[group]
        for user in list_users:
            Z_.append(Z[user].copy())   
        list_Zs.append(Z_)
    return list_Zs

def make_matrix_X_gurobi(list_X_est, G, x1, x2, ls_g1, ls_g2):

    matrix_final = []
    indices = list_X_est[0].index.tolist()
    qtd_g1 = 0
    qtd_g2 = 0

    for i in indices:
        if i in G[1]:
            for j in ls_g1:
                if round(x1['U'+str(i), j].X) == 1:
                    m = list_X_est[int(j[1:]) - 1]
                    matrix_final.append(m.loc[i].tolist())
                    qtd_g1 = qtd_g1 + 1
        if i in G[2]:
            for j in ls_g2:
                if round(x2['U'+str(i), j].X) == 1:
                    m = list_X_est[int(j[1:]) - 1]
                    matrix_final.append(m.loc[i].tolist())
                    qtd_g2 = qtd_g2 + 1

    X_gurobi = pd.DataFrame (matrix_final, columns=list_X_est[0].columns.values)
    X_gurobi['UserID'] = indices
    X_gurobi.set_index("UserID", inplace = True)
    
    return X_gurobi


#--------------------------GUROBI-----------------------------------------------------

# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = False # True: to use MovieLens with more ratings; False: otherwise

# recommendation algorithm
# algorithm = 'RecSysALS'
algorithm = 'RecSysKNN'
# algorithm = 'RecSysNMF'
# algorithm = 'RecSysSGD'
# algorithm = 'RecSysSVD'

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

# X.to_excel('X_MovieLens.xlsx', index=True)
# X_est.to_excel('Xest_MovieLens.xlsx', index=True)

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [before the impartiality algorithm] ------------")

# To capture polarization, we seek to measure the extent to which the user ratings disagree
polarization = Polarization()
Rpol = polarization.evaluate(X_est)
print("Polarization (Rpol):", Rpol)

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

losses = ilv.get_losses(X_est)


list_users = X_est.index.tolist()
advantaged_group = list_users[0:15]
disadvantaged_group = list_users[15:300]
G = {1: advantaged_group, 2: disadvantaged_group}

#G = {1: [424, 549, 889, 1015, 1088, 1150, 1181, 1285, 1449, 1680, 1941, 1980, 2063, 2909, 3391, 3618, 3808, 3841, 4169, 4227, 4277, 4344, 4508, 4510, 5367, 5795, 5831], 2: [48, 53, 123, 148, 149, 173, 202, 216, 245, 302, 308, 319, 329, 331, 411, 438, 509, 528, 533, 543, 660, 692, 699, 721, 731, 770, 839, 855, 877, 984, 1050, 1068, 1069, 1112, 1117, 1120, 1125, 1137, 1194, 1203, 1207, 1224, 1242, 1246, 1264, 1266, 1274, 1298, 1303, 1317, 1333, 1340, 1354, 1422, 1425, 1451, 1465, 1470, 1496, 1579, 1613, 1632, 1635, 1671, 1675, 1737, 1741, 1748, 1749, 1764, 1780, 1812, 1835, 1837, 1884, 1889, 1897, 1899, 1920, 1926, 1943, 1988, 2010, 2012, 2030, 2073, 2077, 2092, 2109, 2124, 2453, 2544, 2793, 2857, 2878, 2887, 2934, 2962, 2986, 3018, 3029, 3118, 3182, 3280, 3285, 3308, 3312, 3320, 3336, 3389, 3410, 3462, 3475, 3476, 3483, 3491, 3519, 3562, 3589, 3610, 3648, 3650, 3681, 3683, 3693, 3705, 3724, 3768, 3821, 3823, 3834, 3842, 3884, 3885, 3929, 3934, 3942, 3999, 4016, 4021, 4033, 4048, 4054, 4083, 4089, 4140, 4186, 4305, 4345, 4354, 4387, 4411, 4482, 4578, 4579, 4682, 4728, 4732, 4802, 4867, 4957, 5011, 5015, 5054, 5074, 5107, 5111, 5220, 5306, 5312, 5387, 5433, 5493, 5501, 5504, 5511, 5530, 5536, 5550, 5605, 5675, 5682, 5747, 5759, 5763, 5878, 5880, 5888, 5916, 5996], 3: [195, 352, 482, 524, 531, 550, 678, 710, 752, 869, 881, 1010, 1019, 1051, 1383, 1447, 1448, 1605, 1647, 1676, 1698, 1733, 1880, 1912, 1958, 2015, 2106, 2116, 2181, 2304, 2507, 2529, 2665, 2777, 2820, 2907, 3032, 3067, 3163, 3224, 3272, 3292, 3311, 3401, 3471, 3507, 3526, 3539, 3626, 3675, 3778, 3792, 3824, 3829, 4041, 4064, 4085, 4238, 4312, 4386, 4425, 4447, 4448, 4543, 4647, 4725, 4808, 4979, 5026, 5046, 5100, 5256, 5333, 5394, 5614, 5627, 5636, 5643, 5788, 5812, 5954, 6016, 6036]}
G_index = {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)


RgrpNR = glv.evaluate(X_est)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)

print(f'Group Loss Variance (Rgrp NR): {RgrpNR:.9f}')
print(f'RMSE: {result:.9f}\n')

#X_est.to_excel('Xest_MovieLens.xlsx', index=True)
#print(X_est)

##############################################################################################################################
algorithmImpartiality = AlgorithmImpartiality(X, omega, 1)
list_X_est = algorithmImpartiality.evaluate(X_est, 10) # calculates a list of 10 estimated matrices

#list_X_est[0].to_excel('X1_MovieLens.xlsx', index=True)

# print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [after the impartiality algorithm] -------------")

# # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# # The loss of group i as the mean squared estimation error over all known ratings in group i
# # G group: identifying the groups (NR: users grouped by number of ratings for available items)
# # Hierarchical clustering (tree clustering - dendrogram)
# G = {1: advantaged_group, 2: disadvantaged_group}
# G_index = {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299]}
# glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)

# for X_est in list_X_est:
#     RgrpNR = glv.evaluate(X_est)
#     print("Group Loss Variance (Rgrp NR):", RgrpNR, end='; ') # NR: cluster by number ratings
#     rmse = RMSE(X, omega)
#     result = rmse.evaluate(X_est)
#     print("RMSE: ", result)

# FRAMEWORK: preparing data for Rgrp minimization
print("\n\n--------------------------------- optimization result ----------------------------------")

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
list_losses = []
for X_est in list_X_est:
    losses = ilv.get_losses(X_est)
    list_losses.append(losses)

Z = losses_to_Z(list_losses)

G = {1: advantaged_group, 2: disadvantaged_group}
G_index = {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299]}
list_Zs = matrices_Zs(Z, G_index)
#list_Zs = matrices_Zs(Z, G)

# User labels and Rindv (individual injustices)
users_g1 = ["U{}".format(user) for user in G[1]]  
ls_g1 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
users_g2 = ["U{}".format(user) for user in G[2]] 
ls_g2 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 



# Dictionary with individual losses
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


# Initialize the model
m = gp.Model()

# Decision variables
x1 = m.addVars(users_g1, ls_g1, vtype=gp.GRB.BINARY)
x2 = m.addVars(users_g2, ls_g2, vtype=gp.GRB.BINARY)


# Objective function
# In this case, the objective function seeks to minimize the variance between the injustices of the groupS (Li) 
# Li can also be understood as the average of the individual injustices of group i. 
# Rgrp: the variance of all the injustices of the groups (Li).

L1 = x1.prod(preferencias1)/len(users_g1)
L2 = x2.prod(preferencias2)/len(users_g2)
LMean = (L1 + L2) / 2
Rgrp = ((L1 - LMean)**2 + (L2 - LMean)**2 )/2

m.setObjective( Rgrp , sense=gp.GRB.MINIMIZE)

# Restrictions that ensure all users will have a Rindv (individual injustice calculation)
c1 = m.addConstrs(
    x1.sum(i, '*') == 1 for i in users_g1)

c2 = m.addConstrs(
    x2.sum(i, '*') == 1 for i in users_g2)


# Run the model
m.optimize()

'''print("Grupo 01")
for i in users_g1:
    for j in ls_g1:
        print("{};".format(round(x1[i, j].X)), end="")
    print("")

print("Grupo 02")
for i in users_g2:
    for j in ls_g2:
        print("{};".format(round(x2[i, j].X)), end="")
    print("")

print("Grupo 03")
for i in users_g3:
    for j in ls_g3:
        print("{};".format(round(x3[i, j].X)), end="")
    print("")'''

# Prints the injustice of the group
#print(f'Rgrp NR: {round(m.objVal):.20f}')

# Calculate the recommendation matrix optimized by gurobi
X_gurobi = make_matrix_X_gurobi(list_X_est, G, x1, x2, ls_g1, ls_g2)

# Show group injustice and optimized recommendation matrix RMSE
RgrpNR = glv.evaluate(X_gurobi)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_gurobi)

print(f'Group Loss Variance (Rgrp NR): {RgrpNR:.9f}')
print(f'RMSE: {result:.9f}\n')

# X_gurobi.to_excel("Xest_MovieLens_Gurobi.xlsx")





