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

# reading data from 20000 books and 30000 users 
Data_path = 'Data/Books'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = False # True: to use books with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysALS'


# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_books(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

X_est.to_excel('Xest_Books.xlsx', index=True)

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
G = {1: [1131, 1211, 1248, 1424, 1435, 1548, 1585, 1674, 1733, 1848, 1903, 2024, 2030, 2033, 2103, 2110, 2179, 2288, 2313, 2337, 2363, 4017, 6242, 6251, 6543, 6575, 7346, 8454, 13552, 16795, 21014, 23872, 23902, 25981, 28177, 28634], 2: [11676], 3: [254, 638, 882, 929, 1155, 1161, 1167, 1178, 1184, 1192, 1214, 1219, 1249, 1254, 1261, 1262, 1279, 1294, 1297, 1331, 1348, 1368, 1372, 1376, 1399, 1409, 1412, 1436, 1466, 1467, 1485, 1499, 1517, 1535, 1554, 1558, 1596, 1597, 1608, 1652, 1660, 1688, 1706, 1719, 1725, 1790, 1791, 1805, 1812, 1830, 1838, 1863, 1869, 1881, 1891, 1898, 1901, 1923, 1928, 1990, 2009, 2010, 2012, 2036, 2041, 2046, 2084, 2090, 2106, 2132, 2134, 2135, 2136, 2139, 2152, 2189, 2197, 2222, 2238, 2255, 2276, 2281, 2287, 2295, 2296, 2326, 2333, 2349, 2354, 2358, 2385, 2389, 2399, 2404, 2406, 2411, 2415, 2437, 2439, 2461, 2462, 2466, 2470, 2481, 2545, 2549, 2552, 2559, 2589, 2597, 2622, 2630, 2644, 2651, 2653, 2678, 2688, 2719, 2766, 2891, 2977, 3145, 3167, 3363, 3371, 3373, 3827, 4221, 4785, 4795, 4938, 5027, 5037, 5582, 5899, 5903, 6073, 6115, 6323, 6532, 6563, 6772, 6789, 7125, 7158, 7283, 7286, 7841, 7915, 8066, 8067, 8187, 8245, 8253, 8680, 8681, 8734, 8930, 9177, 9226, 9856, 9908, 10314, 10447, 10560, 10819, 11224, 11245, 11657, 11718, 11724, 11788, 11944, 12100, 12154, 12489, 12784, 12982, 13080, 13273, 13518, 13551, 13582, 13664, 13666, 13850, 13935, 14422, 14456, 14744, 14768, 15049, 15408, 15418, 15602, 15834, 15957, 16246, 16504, 16599, 16634, 16718, 16916, 16966, 17003, 17859, 17950, 18082, 18203, 19233, 19493, 19664, 19711, 20060, 20106, 20172, 20180, 20462, 20680, 21011, 21031, 21356, 21364, 21404, 21484, 21576, 21659, 22074, 22094, 22252, 22625, 22936, 23547, 23571, 23768, 23933, 24186, 24194, 25131, 25395, 25409, 25410, 25601, 25919, 25966, 26057, 26240, 26371, 26517, 26535, 26538, 26620, 26621, 26883, 27091, 27472, 27740, 28204, 28594, 28647, 29209, 29259, 29526]}
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
G = {1: [1131, 1211, 1248, 1424, 1435, 1548, 1585, 1674, 1733, 1848, 1903, 2024, 2030, 2033, 2103, 2110, 2179, 2288, 2313, 2337, 2363, 4017, 6242, 6251, 6543, 6575, 7346, 8454, 13552, 16795, 21014, 23872, 23902, 25981, 28177, 28634], 2: [11676], 3: [254, 638, 882, 929, 1155, 1161, 1167, 1178, 1184, 1192, 1214, 1219, 1249, 1254, 1261, 1262, 1279, 1294, 1297, 1331, 1348, 1368, 1372, 1376, 1399, 1409, 1412, 1436, 1466, 1467, 1485, 1499, 1517, 1535, 1554, 1558, 1596, 1597, 1608, 1652, 1660, 1688, 1706, 1719, 1725, 1790, 1791, 1805, 1812, 1830, 1838, 1863, 1869, 1881, 1891, 1898, 1901, 1923, 1928, 1990, 2009, 2010, 2012, 2036, 2041, 2046, 2084, 2090, 2106, 2132, 2134, 2135, 2136, 2139, 2152, 2189, 2197, 2222, 2238, 2255, 2276, 2281, 2287, 2295, 2296, 2326, 2333, 2349, 2354, 2358, 2385, 2389, 2399, 2404, 2406, 2411, 2415, 2437, 2439, 2461, 2462, 2466, 2470, 2481, 2545, 2549, 2552, 2559, 2589, 2597, 2622, 2630, 2644, 2651, 2653, 2678, 2688, 2719, 2766, 2891, 2977, 3145, 3167, 3363, 3371, 3373, 3827, 4221, 4785, 4795, 4938, 5027, 5037, 5582, 5899, 5903, 6073, 6115, 6323, 6532, 6563, 6772, 6789, 7125, 7158, 7283, 7286, 7841, 7915, 8066, 8067, 8187, 8245, 8253, 8680, 8681, 8734, 8930, 9177, 9226, 9856, 9908, 10314, 10447, 10560, 10819, 11224, 11245, 11657, 11718, 11724, 11788, 11944, 12100, 12154, 12489, 12784, 12982, 13080, 13273, 13518, 13551, 13582, 13664, 13666, 13850, 13935, 14422, 14456, 14744, 14768, 15049, 15408, 15418, 15602, 15834, 15957, 16246, 16504, 16599, 16634, 16718, 16916, 16966, 17003, 17859, 17950, 18082, 18203, 19233, 19493, 19664, 19711, 20060, 20106, 20172, 20180, 20462, 20680, 21011, 21031, 21356, 21364, 21404, 21484, 21576, 21659, 22074, 22094, 22252, 22625, 22936, 23547, 23571, 23768, 23933, 24186, 24194, 25131, 25395, 25409, 25410, 25601, 25919, 25966, 26057, 26240, 26371, 26517, 26535, 26538, 26620, 26621, 26883, 27091, 27472, 27740, 28204, 28594, 28647, 29209, 29259, 29526]}
G_index = {1: [13, 17, 9, 23, 24, 25, 18, 15, 4, 10, 20, 7, 3, 14, 30, 2, 16, 31, 5, 26, 12, 32, 21, 19, 27, 8, 6, 28, 11, 1, 22, 33, 34, 29, 35, 36], 2: [0], 3: [123, 96, 97, 254, 37, 98, 65, 124, 125, 255, 256, 257, 173, 174, 175, 258, 259, 176, 126, 177, 260, 127, 261, 262, 178, 99, 66, 263, 264, 100, 101, 179, 265, 266, 267, 84, 268, 128, 102, 180, 85, 103, 67, 129, 130, 131, 181, 68, 132, 269, 182, 270, 271, 272, 273, 133, 274, 183, 134, 184, 86, 38, 55, 275, 46, 276, 40, 104, 135, 56, 57, 277, 185, 105, 278, 106, 50, 279, 280, 186, 41, 281, 69, 282, 283, 284, 285, 286, 287, 87, 187, 288, 88, 289, 89, 107, 290, 136, 58, 291, 137, 188, 292, 189, 293, 294, 138, 295, 59, 190, 108, 191, 296, 297, 298, 299, 192, 139, 70, 90, 91, 140, 141, 193, 142, 109, 71, 194, 195, 196, 110, 197, 198, 47, 199, 143, 72, 200, 201, 144, 73, 202, 203, 145, 204, 205, 111, 146, 206, 74, 112, 147, 42, 75, 148, 43, 149, 207, 208, 150, 76, 151, 77, 48, 49, 152, 209, 60, 210, 211, 153, 212, 213, 154, 155, 214, 215, 39, 216, 78, 217, 218, 219, 220, 221, 92, 222, 61, 156, 157, 223, 158, 159, 224, 62, 160, 225, 161, 226, 113, 44, 162, 227, 228, 51, 114, 52, 115, 229, 230, 163, 63, 164, 79, 165, 116, 80, 231, 232, 166, 117, 167, 233, 93, 234, 94, 95, 64, 235, 236, 53, 237, 118, 238, 119, 168, 169, 239, 120, 240, 81, 170, 241, 242, 243, 244, 245, 246, 247, 82, 248, 249, 250, 251, 252, 45, 121, 171, 172, 253, 54, 122, 83]}
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

G = {1: [1131, 1211, 1248, 1424, 1435, 1548, 1585, 1674, 1733, 1848, 1903, 2024, 2030, 2033, 2103, 2110, 2179, 2288, 2313, 2337, 2363, 4017, 6242, 6251, 6543, 6575, 7346, 8454, 13552, 16795, 21014, 23872, 23902, 25981, 28177, 28634], 2: [11676], 3: [254, 638, 882, 929, 1155, 1161, 1167, 1178, 1184, 1192, 1214, 1219, 1249, 1254, 1261, 1262, 1279, 1294, 1297, 1331, 1348, 1368, 1372, 1376, 1399, 1409, 1412, 1436, 1466, 1467, 1485, 1499, 1517, 1535, 1554, 1558, 1596, 1597, 1608, 1652, 1660, 1688, 1706, 1719, 1725, 1790, 1791, 1805, 1812, 1830, 1838, 1863, 1869, 1881, 1891, 1898, 1901, 1923, 1928, 1990, 2009, 2010, 2012, 2036, 2041, 2046, 2084, 2090, 2106, 2132, 2134, 2135, 2136, 2139, 2152, 2189, 2197, 2222, 2238, 2255, 2276, 2281, 2287, 2295, 2296, 2326, 2333, 2349, 2354, 2358, 2385, 2389, 2399, 2404, 2406, 2411, 2415, 2437, 2439, 2461, 2462, 2466, 2470, 2481, 2545, 2549, 2552, 2559, 2589, 2597, 2622, 2630, 2644, 2651, 2653, 2678, 2688, 2719, 2766, 2891, 2977, 3145, 3167, 3363, 3371, 3373, 3827, 4221, 4785, 4795, 4938, 5027, 5037, 5582, 5899, 5903, 6073, 6115, 6323, 6532, 6563, 6772, 6789, 7125, 7158, 7283, 7286, 7841, 7915, 8066, 8067, 8187, 8245, 8253, 8680, 8681, 8734, 8930, 9177, 9226, 9856, 9908, 10314, 10447, 10560, 10819, 11224, 11245, 11657, 11718, 11724, 11788, 11944, 12100, 12154, 12489, 12784, 12982, 13080, 13273, 13518, 13551, 13582, 13664, 13666, 13850, 13935, 14422, 14456, 14744, 14768, 15049, 15408, 15418, 15602, 15834, 15957, 16246, 16504, 16599, 16634, 16718, 16916, 16966, 17003, 17859, 17950, 18082, 18203, 19233, 19493, 19664, 19711, 20060, 20106, 20172, 20180, 20462, 20680, 21011, 21031, 21356, 21364, 21404, 21484, 21576, 21659, 22074, 22094, 22252, 22625, 22936, 23547, 23571, 23768, 23933, 24186, 24194, 25131, 25395, 25409, 25410, 25601, 25919, 25966, 26057, 26240, 26371, 26517, 26535, 26538, 26620, 26621, 26883, 27091, 27472, 27740, 28204, 28594, 28647, 29209, 29259, 29526]}
G_index = {1: [13, 17, 9, 23, 24, 25, 18, 15, 4, 10, 20, 7, 3, 14, 30, 2, 16, 31, 5, 26, 12, 32, 21, 19, 27, 8, 6, 28, 11, 1, 22, 33, 34, 29, 35, 36], 2: [0], 3: [123, 96, 97, 254, 37, 98, 65, 124, 125, 255, 256, 257, 173, 174, 175, 258, 259, 176, 126, 177, 260, 127, 261, 262, 178, 99, 66, 263, 264, 100, 101, 179, 265, 266, 267, 84, 268, 128, 102, 180, 85, 103, 67, 129, 130, 131, 181, 68, 132, 269, 182, 270, 271, 272, 273, 133, 274, 183, 134, 184, 86, 38, 55, 275, 46, 276, 40, 104, 135, 56, 57, 277, 185, 105, 278, 106, 50, 279, 280, 186, 41, 281, 69, 282, 283, 284, 285, 286, 287, 87, 187, 288, 88, 289, 89, 107, 290, 136, 58, 291, 137, 188, 292, 189, 293, 294, 138, 295, 59, 190, 108, 191, 296, 297, 298, 299, 192, 139, 70, 90, 91, 140, 141, 193, 142, 109, 71, 194, 195, 196, 110, 197, 198, 47, 199, 143, 72, 200, 201, 144, 73, 202, 203, 145, 204, 205, 111, 146, 206, 74, 112, 147, 42, 75, 148, 43, 149, 207, 208, 150, 76, 151, 77, 48, 49, 152, 209, 60, 210, 211, 153, 212, 213, 154, 155, 214, 215, 39, 216, 78, 217, 218, 219, 220, 221, 92, 222, 61, 156, 157, 223, 158, 159, 224, 62, 160, 225, 161, 226, 113, 44, 162, 227, 228, 51, 114, 52, 115, 229, 230, 163, 63, 164, 79, 165, 116, 80, 231, 232, 166, 117, 167, 233, 93, 234, 94, 95, 64, 235, 236, 53, 237, 118, 238, 119, 168, 169, 239, 120, 240, 81, 170, 241, 242, 243, 244, 245, 246, 247, 82, 248, 249, 250, 251, 252, 45, 121, 171, 172, 253, 54, 122, 83]}
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

X_gurobi.to_excel("Xest_Books_Gurobi.xlsx")





