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

def make_matrix_X_gurobi(list_X_est, G, x1, x2, x3, ls_g1, ls_g2, ls_g3):

    matrix_final = []
    indices = list_X_est[0].index.tolist()
    qtd_g1 = 0
    qtd_g2 = 0
    qtd_g3 = 0

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
        if i in G[3]:
            for j in ls_g3:
                if round(x3['U'+str(i), j].X) == 1:
                    m = list_X_est[int(j[1:]) - 1]
                    matrix_final.append(m.loc[i].tolist())
                    qtd_g3 = qtd_g3 + 1

    X_gurobi = pd.DataFrame (matrix_final, columns=list_X_est[0].columns.values)
    X_gurobi['UserID'] = indices
    X_gurobi.set_index("UserID", inplace = True)
    
    return X_gurobi


#--------------------------GUROBI-----------------------------------------------------

# reading data from 19993 songs and 16000 users 
Data_path = 'Data/Songs'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use songs with more ratings; False: otherwise

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

X, users_info, items_info = recsys.read_songs(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

X_est.to_excel('Xest_Songs.xlsx', index=True)

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

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# Hierarchical clustering (tree clustering - dendrogram)
G = {1: [18, 50, 54, 114, 132, 133, 177, 185, 193, 195, 204, 208, 303, 324, 348, 352, 462, 516, 527, 549, 561, 572, 575, 589, 599, 623, 628, 630, 764, 775, 813, 814, 817, 826, 832, 886, 925, 954, 977, 999, 1005, 1008, 1042, 1047, 1109, 1194, 1199, 1202, 1259, 1296, 1359, 1376, 1394, 1396, 1398, 1437, 1479, 1483, 1489, 1563, 1584, 1588, 1611, 1633, 1702, 1712, 1757, 1787, 1819, 1821, 1824, 1828, 1835, 1904, 1912, 1938, 1946, 1980, 1983, 2046, 2051, 2060, 2123, 2162, 2164, 2183, 2187, 2199, 2227, 2230, 2262, 2270, 2295, 2297, 2327, 2350, 2353, 2354, 2384, 2401, 2403, 2460, 2519, 2614, 2644, 2695, 2702, 2784, 2863, 2881, 2895, 2910, 2931, 2952, 2954, 2964, 2971, 2973, 3025, 3037, 3117, 3149, 3153, 3237, 3374, 3383, 3438, 3447, 3474, 3500, 3506, 3519, 3534, 3553, 3658, 3684, 3839, 3893, 3964, 4040, 4138, 4339, 4376, 4440, 4515, 4676, 4789, 5082, 5110, 5159, 5489, 5579, 5673, 5832, 5908, 6024, 6176, 6204, 6224, 6587, 6624, 6810, 6931, 7057, 7284, 7297, 7494, 7547, 7669, 7786, 8081, 8109, 8189, 8236, 8569, 8636, 8960, 8962, 9111, 9116, 9134, 9135, 9224, 9334, 9378, 9443, 9513, 9620, 9698, 9719, 9792, 9879, 9990, 9999, 10029, 10152, 10525, 10605, 10646, 10664, 10860, 10893, 10931, 11119, 11306, 11428, 11460, 11714, 11938, 11978, 12138, 12169, 12941, 12965, 13181, 13361, 13474, 13548, 13750, 13773, 13911, 13959, 14276, 14419, 14491, 14615, 14694, 15313, 15346, 15352, 15472, 15475, 15611, 15714, 15779, 15836], 2: [152, 167, 199, 441, 460, 670, 724, 784, 1136, 1228, 1548, 2094, 2197, 2390, 2521, 2766, 2792, 3044, 3085, 3130, 3154, 3599, 3808, 4360, 4973, 5938, 6088, 6336, 9856, 11440, 13335, 14621, 15293, 15771], 3: [166, 270, 405, 650, 1262, 1328, 1426, 1499, 1762, 1816, 2023, 2233, 2451, 2455, 2573, 2607, 2864, 3523, 3834, 3891, 4114, 6405, 6441, 7324, 9766, 10204, 11968, 13949, 15604, 15759]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
RgrpNR = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NR):", RgrpNR) # NR: cluster by number ratings

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)
print("RMSE: ", result)


##############################################################################################################################
algorithmImpartiality = AlgorithmImpartiality(X, omega, 1)
list_X_est = algorithmImpartiality.evaluate(X_est, 15) # calculates a list of 10 estimated matrices

#list_X_est[0].to_excel('X1_MovieLens.xlsx', index=True)

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [after the impartiality algorithm] -------------")

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# Hierarchical clustering (tree clustering - dendrogram)
G = {1: [18, 50, 54, 114, 132, 133, 177, 185, 193, 195, 204, 208, 303, 324, 348, 352, 462, 516, 527, 549, 561, 572, 575, 589, 599, 623, 628, 630, 764, 775, 813, 814, 817, 826, 832, 886, 925, 954, 977, 999, 1005, 1008, 1042, 1047, 1109, 1194, 1199, 1202, 1259, 1296, 1359, 1376, 1394, 1396, 1398, 1437, 1479, 1483, 1489, 1563, 1584, 1588, 1611, 1633, 1702, 1712, 1757, 1787, 1819, 1821, 1824, 1828, 1835, 1904, 1912, 1938, 1946, 1980, 1983, 2046, 2051, 2060, 2123, 2162, 2164, 2183, 2187, 2199, 2227, 2230, 2262, 2270, 2295, 2297, 2327, 2350, 2353, 2354, 2384, 2401, 2403, 2460, 2519, 2614, 2644, 2695, 2702, 2784, 2863, 2881, 2895, 2910, 2931, 2952, 2954, 2964, 2971, 2973, 3025, 3037, 3117, 3149, 3153, 3237, 3374, 3383, 3438, 3447, 3474, 3500, 3506, 3519, 3534, 3553, 3658, 3684, 3839, 3893, 3964, 4040, 4138, 4339, 4376, 4440, 4515, 4676, 4789, 5082, 5110, 5159, 5489, 5579, 5673, 5832, 5908, 6024, 6176, 6204, 6224, 6587, 6624, 6810, 6931, 7057, 7284, 7297, 7494, 7547, 7669, 7786, 8081, 8109, 8189, 8236, 8569, 8636, 8960, 8962, 9111, 9116, 9134, 9135, 9224, 9334, 9378, 9443, 9513, 9620, 9698, 9719, 9792, 9879, 9990, 9999, 10029, 10152, 10525, 10605, 10646, 10664, 10860, 10893, 10931, 11119, 11306, 11428, 11460, 11714, 11938, 11978, 12138, 12169, 12941, 12965, 13181, 13361, 13474, 13548, 13750, 13773, 13911, 13959, 14276, 14419, 14491, 14615, 14694, 15313, 15346, 15352, 15472, 15475, 15611, 15714, 15779, 15836], 2: [152, 167, 199, 441, 460, 670, 724, 784, 1136, 1228, 1548, 2094, 2197, 2390, 2521, 2766, 2792, 3044, 3085, 3130, 3154, 3599, 3808, 4360, 4973, 5938, 6088, 6336, 9856, 11440, 13335, 14621, 15293, 15771], 3: [166, 270, 405, 650, 1262, 1328, 1426, 1499, 1762, 1816, 2023, 2233, 2451, 2455, 2573, 2607, 2864, 3523, 3834, 3891, 4114, 6405, 6441, 7324, 9766, 10204, 11968, 13949, 15604, 15759]}
G_index = {1: [166, 174, 168, 169, 170, 171, 175, 176, 177, 178, 24, 25, 181, 182, 183, 26, 185, 186, 29, 4, 30, 187, 188, 189, 190, 191, 192, 193, 195, 196, 198, 33, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 34, 213, 214, 216, 35, 219, 220, 221, 222, 223, 225, 226, 227, 228, 231, 232, 233, 234, 235, 236, 237, 238, 240, 36, 242, 243, 244, 245, 246, 247, 248, 249, 250, 37, 251, 252, 253, 254, 255, 40, 256, 257, 259, 260, 261, 41, 263, 264, 265, 266, 42, 43, 267, 268, 270, 271, 274, 44, 278, 279, 280, 281, 282, 284, 286, 287, 288, 289, 290, 291, 45, 6, 292, 293, 294, 296, 297, 298, 47, 48, 49, 174, 51, 52, 53, 168, 55, 57, 58, 60, 1, 63, 8, 65, 66, 68, 69, 71, 72, 73, 2, 74, 76, 77, 78, 79, 80, 81, 9, 82, 84, 86, 87, 10, 91, 92, 93, 94, 95, 96, 11, 98, 99, 100, 101, 102, 103, 104, 12, 105, 106, 107, 108, 109, 110, 111, 112, 13, 113, 169, 14, 115, 116, 15, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 173, 129, 19, 130, 131, 171, 134, 135, 137, 138, 139, 20, 140, 141, 143, 144, 21, 145, 146, 147, 3, 149, 150, 151, 172, 154, 156, 157, 158, 159, 160, 162, 22, 23, 165], 2: [172, 174, 179, 28, 184, 31, 32, 197, 212, 215, 230, 39, 258, 269, 275, 5, 283, 295, 46, 7, 299, 59, 61, 70, 75, 83, 85, 88, 0, 170, 142, 153, 155, 164], 3: [173, 180, 27, 194, 217, 218, 224, 229, 239, 241, 38, 262, 272, 273, 276, 277, 285, 56, 62, 64, 67, 89, 90, 97, 16, 17, 136, 148, 161, 163]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)

for X_est in list_X_est:
    RgrpNR = glv.evaluate(X_est)
    print("Group Loss Variance (Rgrp NR):", RgrpNR, end='; ') # NR: cluster by number ratings
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

G = {1: [18, 50, 54, 114, 132, 133, 177, 185, 193, 195, 204, 208, 303, 324, 348, 352, 462, 516, 527, 549, 561, 572, 575, 589, 599, 623, 628, 630, 764, 775, 813, 814, 817, 826, 832, 886, 925, 954, 977, 999, 1005, 1008, 1042, 1047, 1109, 1194, 1199, 1202, 1259, 1296, 1359, 1376, 1394, 1396, 1398, 1437, 1479, 1483, 1489, 1563, 1584, 1588, 1611, 1633, 1702, 1712, 1757, 1787, 1819, 1821, 1824, 1828, 1835, 1904, 1912, 1938, 1946, 1980, 1983, 2046, 2051, 2060, 2123, 2162, 2164, 2183, 2187, 2199, 2227, 2230, 2262, 2270, 2295, 2297, 2327, 2350, 2353, 2354, 2384, 2401, 2403, 2460, 2519, 2614, 2644, 2695, 2702, 2784, 2863, 2881, 2895, 2910, 2931, 2952, 2954, 2964, 2971, 2973, 3025, 3037, 3117, 3149, 3153, 3237, 3374, 3383, 3438, 3447, 3474, 3500, 3506, 3519, 3534, 3553, 3658, 3684, 3839, 3893, 3964, 4040, 4138, 4339, 4376, 4440, 4515, 4676, 4789, 5082, 5110, 5159, 5489, 5579, 5673, 5832, 5908, 6024, 6176, 6204, 6224, 6587, 6624, 6810, 6931, 7057, 7284, 7297, 7494, 7547, 7669, 7786, 8081, 8109, 8189, 8236, 8569, 8636, 8960, 8962, 9111, 9116, 9134, 9135, 9224, 9334, 9378, 9443, 9513, 9620, 9698, 9719, 9792, 9879, 9990, 9999, 10029, 10152, 10525, 10605, 10646, 10664, 10860, 10893, 10931, 11119, 11306, 11428, 11460, 11714, 11938, 11978, 12138, 12169, 12941, 12965, 13181, 13361, 13474, 13548, 13750, 13773, 13911, 13959, 14276, 14419, 14491, 14615, 14694, 15313, 15346, 15352, 15472, 15475, 15611, 15714, 15779, 15836], 2: [152, 167, 199, 441, 460, 670, 724, 784, 1136, 1228, 1548, 2094, 2197, 2390, 2521, 2766, 2792, 3044, 3085, 3130, 3154, 3599, 3808, 4360, 4973, 5938, 6088, 6336, 9856, 11440, 13335, 14621, 15293, 15771], 3: [166, 270, 405, 650, 1262, 1328, 1426, 1499, 1762, 1816, 2023, 2233, 2451, 2455, 2573, 2607, 2864, 3523, 3834, 3891, 4114, 6405, 6441, 7324, 9766, 10204, 11968, 13949, 15604, 15759]}
G_index = {1: [166, 174, 168, 169, 170, 171, 175, 176, 177, 178, 24, 25, 181, 182, 183, 26, 185, 186, 29, 4, 30, 187, 188, 189, 190, 191, 192, 193, 195, 196, 198, 33, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 34, 213, 214, 216, 35, 219, 220, 221, 222, 223, 225, 226, 227, 228, 231, 232, 233, 234, 235, 236, 237, 238, 240, 36, 242, 243, 244, 245, 246, 247, 248, 249, 250, 37, 251, 252, 253, 254, 255, 40, 256, 257, 259, 260, 261, 41, 263, 264, 265, 266, 42, 43, 267, 268, 270, 271, 274, 44, 278, 279, 280, 281, 282, 284, 286, 287, 288, 289, 290, 291, 45, 6, 292, 293, 294, 296, 297, 298, 47, 48, 49, 174, 51, 52, 53, 168, 55, 57, 58, 60, 1, 63, 8, 65, 66, 68, 69, 71, 72, 73, 2, 74, 76, 77, 78, 79, 80, 81, 9, 82, 84, 86, 87, 10, 91, 92, 93, 94, 95, 96, 11, 98, 99, 100, 101, 102, 103, 104, 12, 105, 106, 107, 108, 109, 110, 111, 112, 13, 113, 169, 14, 115, 116, 15, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 173, 129, 19, 130, 131, 171, 134, 135, 137, 138, 139, 20, 140, 141, 143, 144, 21, 145, 146, 147, 3, 149, 150, 151, 172, 154, 156, 157, 158, 159, 160, 162, 22, 23, 165], 2: [172, 174, 179, 28, 184, 31, 32, 197, 212, 215, 230, 39, 258, 269, 275, 5, 283, 295, 46, 7, 299, 59, 61, 70, 75, 83, 85, 88, 0, 170, 142, 153, 155, 164], 3: [173, 180, 27, 194, 217, 218, 224, 229, 239, 241, 38, 262, 272, 273, 276, 277, 285, 56, 62, 64, 67, 89, 90, 97, 16, 17, 136, 148, 161, 163]}
list_Zs = matrices_Zs(Z, G_index)
#list_Zs = matrices_Zs(Z, G)

# User labels and Rindv (individual injustices)
users_g1 = ["U{}".format(user) for user in G[1]]  
ls_g1 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
users_g2 = ["U{}".format(user) for user in G[2]] 
ls_g2 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
users_g3 = ["U{}".format(user) for user in G[3]] 
ls_g3 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 



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

Z3 = list_Zs[2]
preferencias3 = dict()
for i, user in enumerate(users_g3):
    for j, l in enumerate(ls_g3):
        preferencias3[user, l] = Z3[i][j]

# Initialize the model
m = gp.Model()

# Decision variables
x1 = m.addVars(users_g1, ls_g1, vtype=gp.GRB.BINARY)
x2 = m.addVars(users_g2, ls_g2, vtype=gp.GRB.BINARY)
x3 = m.addVars(users_g3, ls_g3, vtype=gp.GRB.BINARY)


# Objective function
# In this case, the objective function seeks to minimize the variance between the injustices of the groupS (Li) 
# Li can also be understood as the average of the individual injustices of group i. 
# Rgrp: the variance of all the injustices of the groups (Li).

L1 = x1.prod(preferencias1)/len(users_g1)
L2 = x2.prod(preferencias2)/len(users_g2)
L3 = x3.prod(preferencias3)/len(users_g3)
LMean = (L1 + L2 + L3) / 2
Rgrp = ((L1 - LMean)**2 + (L2 - LMean)**2 + (L3 - LMean)**2)/3

m.setObjective( Rgrp , sense=gp.GRB.MINIMIZE)

# Restrictions that ensure all users will have a Rindv (individual injustice calculation)
c1 = m.addConstrs(
    x1.sum(i, '*') == 1 for i in users_g1)

c2 = m.addConstrs(
    x2.sum(i, '*') == 1 for i in users_g2)

c3 = m.addConstrs(
    x3.sum(i, '*') == 1 for i in users_g3)

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
X_gurobi = make_matrix_X_gurobi(list_X_est, G, x1, x2, x3, ls_g1, ls_g2, ls_g3)

# Show group injustice and optimized recommendation matrix RMSE
RgrpNR = glv.evaluate(X_gurobi)
print("Group Loss Variance (Rgrp NR):", RgrpNR, end='; ')

rmse = RMSE(X, omega)
result = rmse.evaluate(X_gurobi)
print("RMSE: ", result)

X_gurobi.to_excel("Xest_Songs_Gurobi.xlsx")





