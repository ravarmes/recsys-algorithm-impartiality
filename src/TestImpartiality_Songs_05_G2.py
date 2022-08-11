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

# reading data from 19993 songs and 16000 users 
Data_path = 'Data/Songs'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use songs with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysALS'


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

X_est.to_excel('X_est_Songs.xlsx', index=True)
#print(X_est)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# 5-95%
G = {1: [9856, 3684, 4676, 13959, 549, 2766, 2971, 3130, 3893, 5832, 6224, 7297, 8236, 9224, 9443], 2: [9698, 9766, 10204, 10893, 11119, 12941, 13548, 15714, 15779, 204, 208, 352, 405, 441, 527, 561, 670, 724, 814, 1194, 1296, 1819, 1983, 2023, 2094, 2164, 2262, 2350, 2353, 2519, 2964, 3085, 3237, 3374, 3383, 3438, 3447, 3474, 3500, 3506, 3519, 3523, 3534, 3553, 3599, 3658, 3808, 3834, 3839, 3891, 3964, 4040, 4114, 4138, 4339, 4360, 4376, 4440, 4515, 4789, 4973, 5082, 5110, 5159, 5489, 5579, 5673, 5908, 5938, 6024, 6088, 6176, 6204, 6336, 6405, 6441, 6587, 6624, 6810, 6931, 7057, 7284, 7324, 7494, 7547, 7669, 7786, 8081, 8109, 8189, 8569, 8636, 8960, 8962, 9111, 9116, 9134, 9135, 9334, 9378, 9513, 9620, 9719, 9792, 9879, 9990, 9999, 10029, 10152, 10525, 10605, 10646, 10664, 10860, 10931, 11306, 11428, 11440, 11460, 11714, 11938, 11968, 11978, 12138, 12169, 12965, 13181, 13335, 13361, 13474, 13750, 13773, 13911, 13949, 14276, 14419, 14491, 14615, 14621, 14694, 15293, 15313, 15346, 15352, 15472, 15475, 15604, 15611, 15759, 15771, 15836, 18, 50, 54, 114, 132, 133, 152, 166, 167, 177, 185, 193, 195, 199, 270, 303, 324, 348, 460, 462, 516, 572, 575, 589, 599, 623, 628, 630, 650, 764, 775, 784, 813, 817, 826, 832, 886, 925, 954, 977, 999, 1005, 1008, 1042, 1047, 1109, 1136, 1199, 1202, 1228, 1259, 1262, 1328, 1359, 1376, 1394, 1396, 1398, 1426, 1437, 1479, 1483, 1489, 1499, 1548, 1563, 1584, 1588, 1611, 1633, 1702, 1712, 1757, 1762, 1787, 1816, 1821, 1824, 1828, 1835, 1904, 1912, 1938, 1946, 1980, 2046, 2051, 2060, 2123, 2162, 2183, 2187, 2197, 2199, 2227, 2230, 2233, 2270, 2295, 2297, 2327, 2354, 2384, 2390, 2401, 2403, 2451, 2455, 2460, 2521, 2573, 2607, 2614, 2644, 2695, 2702, 2784, 2792, 2863, 2864, 2881, 2895, 2910, 2931, 2952, 2954, 2973, 3025, 3037, 3044, 3117, 3149, 3153, 3154]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
RgrpNR = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NR):", RgrpNR) # NR: cluster by number ratings

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)
print("RMSE: ", result)


##############################################################################################################################
algorithmImpartiality = AlgorithmImpartiality(X, omega, 1)
list_X_est = algorithmImpartiality.evaluate(X_est, 5) # calculates a list of 10 estimated matrices

#list_X_est[0].to_excel('X1_Songs.xlsx', index=True)

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [after the impartiality algorithm] -------------")

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NR: users grouped by number of ratings for available items)
# Hierarchical clustering (tree clustering - dendrogram)
G = {1: [9856, 3684, 4676, 13959, 549, 2766, 2971, 3130, 3893, 5832, 6224, 7297, 8236, 9224, 9443], 2: [9698, 9766, 10204, 10893, 11119, 12941, 13548, 15714, 15779, 204, 208, 352, 405, 441, 527, 561, 670, 724, 814, 1194, 1296, 1819, 1983, 2023, 2094, 2164, 2262, 2350, 2353, 2519, 2964, 3085, 3237, 3374, 3383, 3438, 3447, 3474, 3500, 3506, 3519, 3523, 3534, 3553, 3599, 3658, 3808, 3834, 3839, 3891, 3964, 4040, 4114, 4138, 4339, 4360, 4376, 4440, 4515, 4789, 4973, 5082, 5110, 5159, 5489, 5579, 5673, 5908, 5938, 6024, 6088, 6176, 6204, 6336, 6405, 6441, 6587, 6624, 6810, 6931, 7057, 7284, 7324, 7494, 7547, 7669, 7786, 8081, 8109, 8189, 8569, 8636, 8960, 8962, 9111, 9116, 9134, 9135, 9334, 9378, 9513, 9620, 9719, 9792, 9879, 9990, 9999, 10029, 10152, 10525, 10605, 10646, 10664, 10860, 10931, 11306, 11428, 11440, 11460, 11714, 11938, 11968, 11978, 12138, 12169, 12965, 13181, 13335, 13361, 13474, 13750, 13773, 13911, 13949, 14276, 14419, 14491, 14615, 14621, 14694, 15293, 15313, 15346, 15352, 15472, 15475, 15604, 15611, 15759, 15771, 15836, 18, 50, 54, 114, 132, 133, 152, 166, 167, 177, 185, 193, 195, 199, 270, 303, 324, 348, 460, 462, 516, 572, 575, 589, 599, 623, 628, 630, 650, 764, 775, 784, 813, 817, 826, 832, 886, 925, 954, 977, 999, 1005, 1008, 1042, 1047, 1109, 1136, 1199, 1202, 1228, 1259, 1262, 1328, 1359, 1376, 1394, 1396, 1398, 1426, 1437, 1479, 1483, 1489, 1499, 1548, 1563, 1584, 1588, 1611, 1633, 1702, 1712, 1757, 1762, 1787, 1816, 1821, 1824, 1828, 1835, 1904, 1912, 1938, 1946, 1980, 2046, 2051, 2060, 2123, 2162, 2183, 2187, 2197, 2199, 2227, 2230, 2233, 2270, 2295, 2297, 2327, 2354, 2384, 2390, 2401, 2403, 2451, 2455, 2460, 2521, 2573, 2607, 2614, 2644, 2695, 2702, 2784, 2792, 2863, 2864, 2881, 2895, 2910, 2931, 2952, 2954, 2973, 3025, 3037, 3044, 3117, 3149, 3153, 3154]}
G_index = {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299]}
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

G = {1: [9856, 3684, 4676, 13959, 549, 2766, 2971, 3130, 3893, 5832, 6224, 7297, 8236, 9224, 9443], 2: [9698, 9766, 10204, 10893, 11119, 12941, 13548, 15714, 15779, 204, 208, 352, 405, 441, 527, 561, 670, 724, 814, 1194, 1296, 1819, 1983, 2023, 2094, 2164, 2262, 2350, 2353, 2519, 2964, 3085, 3237, 3374, 3383, 3438, 3447, 3474, 3500, 3506, 3519, 3523, 3534, 3553, 3599, 3658, 3808, 3834, 3839, 3891, 3964, 4040, 4114, 4138, 4339, 4360, 4376, 4440, 4515, 4789, 4973, 5082, 5110, 5159, 5489, 5579, 5673, 5908, 5938, 6024, 6088, 6176, 6204, 6336, 6405, 6441, 6587, 6624, 6810, 6931, 7057, 7284, 7324, 7494, 7547, 7669, 7786, 8081, 8109, 8189, 8569, 8636, 8960, 8962, 9111, 9116, 9134, 9135, 9334, 9378, 9513, 9620, 9719, 9792, 9879, 9990, 9999, 10029, 10152, 10525, 10605, 10646, 10664, 10860, 10931, 11306, 11428, 11440, 11460, 11714, 11938, 11968, 11978, 12138, 12169, 12965, 13181, 13335, 13361, 13474, 13750, 13773, 13911, 13949, 14276, 14419, 14491, 14615, 14621, 14694, 15293, 15313, 15346, 15352, 15472, 15475, 15604, 15611, 15759, 15771, 15836, 18, 50, 54, 114, 132, 133, 152, 166, 167, 177, 185, 193, 195, 199, 270, 303, 324, 348, 460, 462, 516, 572, 575, 589, 599, 623, 628, 630, 650, 764, 775, 784, 813, 817, 826, 832, 886, 925, 954, 977, 999, 1005, 1008, 1042, 1047, 1109, 1136, 1199, 1202, 1228, 1259, 1262, 1328, 1359, 1376, 1394, 1396, 1398, 1426, 1437, 1479, 1483, 1489, 1499, 1548, 1563, 1584, 1588, 1611, 1633, 1702, 1712, 1757, 1762, 1787, 1816, 1821, 1824, 1828, 1835, 1904, 1912, 1938, 1946, 1980, 2046, 2051, 2060, 2123, 2162, 2183, 2187, 2197, 2199, 2227, 2230, 2233, 2270, 2295, 2297, 2327, 2354, 2384, 2390, 2401, 2403, 2451, 2455, 2460, 2521, 2573, 2607, 2614, 2644, 2695, 2702, 2784, 2792, 2863, 2864, 2881, 2895, 2910, 2931, 2952, 2954, 2973, 3025, 3037, 3044, 3117, 3149, 3153, 3154]}
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
print("Group Loss Variance (Rgrp NR):", RgrpNR, end='; ')

rmse = RMSE(X, omega)
result = rmse.evaluate(X_gurobi)
print("RMSE: ", result)

X_gurobi.to_excel("Xest_Songs_Gurobi.xlsx")





