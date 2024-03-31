from RecSys import RecSys
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance


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

X, genres, user_info = recsys.read_songs(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_books columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS ------------")

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

# G group: identifying the groups (NA: users grouped by number of ratings for available items)
# advantaged group: 5% users with the highest number of item ratings
# disadvantaged group: 95% users with the lowest number of item ratings
list_users = X_est.index.tolist()
advantaged_group = list_users[0:15]
disadvantaged_group = list_users[15:300]
G1 = {1: advantaged_group, 2: disadvantaged_group}

glv = GroupLossVariance(X, omega, G1, 1) #axis = 1 (0 rows e 1 columns)
RgrpNA = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NA):", RgrpNA)


# G group: identifying the groups (IU: individual unfairness - the variance of the user losses)
# The configuration of groups was based in the hierarchical clustering (tree clustering - dendrogram)
# Clusters 1, 2 and 3
G2 = {1: [18, 50, 54, 114, 132, 133, 177, 185, 193, 195, 204, 208, 303, 324, 348, 352, 462, 516, 527, 549, 561, 572, 575, 589, 599, 623, 628, 630, 764, 775, 813, 814, 817, 826, 832, 886, 925, 954, 977, 999, 1005, 1008, 1042, 1047, 1109, 1194, 1199, 1202, 1259, 1296, 1359, 1376, 1394, 1396, 1398, 1437, 1479, 1483, 1489, 1563, 1584, 1588, 1611, 1633, 1702, 1712, 1757, 1787, 1819, 1821, 1824, 1828, 1835, 1904, 1912, 1938, 1946, 1980, 1983, 2046, 2051, 2060, 2123, 2162, 2164, 2183, 2187, 2199, 2227, 2230, 2262, 2270, 2295, 2297, 2327, 2350, 2353, 2354, 2384, 2401, 2403, 2460, 2519, 2614, 2644, 2695, 2702, 2784, 2863, 2881, 2895, 2910, 2931, 2952, 2954, 2964, 2971, 2973, 3025, 3037, 3117, 3149, 3153, 3237, 3374, 3383, 3438, 3447, 3474, 3500, 3506, 3519, 3534, 3553, 3658, 3684, 3839, 3893, 3964, 4040, 4138, 4339, 4376, 4440, 4515, 4676, 4789, 5082, 5110, 5159, 5489, 5579, 5673, 5832, 5908, 6024, 6176, 6204, 6224, 6587, 6624, 6810, 6931, 7057, 7284, 7297, 7494, 7547, 7669, 7786, 8081, 8109, 8189, 8236, 8569, 8636, 8960, 8962, 9111, 9116, 9134, 9135, 9224, 9334, 9378, 9443, 9513, 9620, 9698, 9719, 9792, 9879, 9990, 9999, 10029, 10152, 10525, 10605, 10646, 10664, 10860, 10893, 10931, 11119, 11306, 11428, 11460, 11714, 11938, 11978, 12138, 12169, 12941, 12965, 13181, 13361, 13474, 13548, 13750, 13773, 13911, 13959, 14276, 14419, 14491, 14615, 14694, 15313, 15346, 15352, 15472, 15475, 15611, 15714, 15779, 15836], 2: [152, 167, 199, 441, 460, 670, 724, 784, 1136, 1228, 1548, 2094, 2197, 2390, 2521, 2766, 2792, 3044, 3085, 3130, 3154, 3599, 3808, 4360, 4973, 5938, 6088, 6336, 9856, 11440, 13335, 14621, 15293, 15771], 3: [166, 270, 405, 650, 1262, 1328, 1426, 1499, 1762, 1816, 2023, 2233, 2451, 2455, 2573, 2607, 2864, 3523, 3834, 3891, 4114, 6405, 6441, 7324, 9766, 10204, 11968, 13949, 15604, 15759]}

glv = GroupLossVariance(X, omega, G2, 1) #axis = 1 (0 rows e 1 columns)
RgrpIU = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp IU):", RgrpIU)