from RecSys import RecSys
from AlgorithmUserFairness import RMSE, Polarization, IndividualLossVariance, GroupLossVariance
from AlgorithmImpartiality import AlgorithmImpartiality

# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = False # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysALS'

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [before the impartiality algorithm] ------------")

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i

# G group: identifying the groups (NR: number ratings)
# The configuration of groups was based in the hierarchical clustering (tree clustering - dendrogram)
# Clusters 1, 2 and 3
G2 = {1: [424, 549, 889, 1015, 1088, 1150, 1181, 1285, 1449, 1680, 1941, 1980, 2063, 2909, 3391, 3618, 3808, 3841, 4169, 4227, 4277, 4344, 4508, 4510, 5367, 5795, 5831], 2: [48, 53, 123, 148, 149, 173, 202, 216, 245, 302, 308, 319, 329, 331, 411, 438, 509, 528, 533, 543, 660, 692, 699, 721, 731, 770, 839, 855, 877, 984, 1050, 1068, 1069, 1112, 1117, 1120, 1125, 1137, 1194, 1203, 1207, 1224, 1242, 1246, 1264, 1266, 1274, 1298, 1303, 1317, 1333, 1340, 1354, 1422, 1425, 1451, 1465, 1470, 1496, 1579, 1613, 1632, 1635, 1671, 1675, 1737, 1741, 1748, 1749, 1764, 1780, 1812, 1835, 1837, 1884, 1889, 1897, 1899, 1920, 1926, 1943, 1988, 2010, 2012, 2030, 2073, 2077, 2092, 2109, 2124, 2453, 2544, 2793, 2857, 2878, 2887, 2934, 2962, 2986, 3018, 3029, 3118, 3182, 3280, 3285, 3308, 3312, 3320, 3336, 3389, 3410, 3462, 3475, 3476, 3483, 3491, 3519, 3562, 3589, 3610, 3648, 3650, 3681, 3683, 3693, 3705, 3724, 3768, 3821, 3823, 3834, 3842, 3884, 3885, 3929, 3934, 3942, 3999, 4016, 4021, 4033, 4048, 4054, 4083, 4089, 4140, 4186, 4305, 4345, 4354, 4387, 4411, 4482, 4578, 4579, 4682, 4728, 4732, 4802, 4867, 4957, 5011, 5015, 5054, 5074, 5107, 5111, 5220, 5306, 5312, 5387, 5433, 5493, 5501, 5504, 5511, 5530, 5536, 5550, 5605, 5675, 5682, 5747, 5759, 5763, 5878, 5880, 5888, 5916, 5996], 3: [195, 352, 482, 524, 531, 550, 678, 710, 752, 869, 881, 1010, 1019, 1051, 1383, 1447, 1448, 1605, 1647, 1676, 1698, 1733, 1880, 1912, 1958, 2015, 2106, 2116, 2181, 2304, 2507, 2529, 2665, 2777, 2820, 2907, 3032, 3067, 3163, 3224, 3272, 3292, 3311, 3401, 3471, 3507, 3526, 3539, 3626, 3675, 3778, 3792, 3824, 3829, 4041, 4064, 4085, 4238, 4312, 4386, 4425, 4447, 4448, 4543, 4647, 4725, 4808, 4979, 5026, 5046, 5100, 5256, 5333, 5394, 5614, 5627, 5636, 5643, 5788, 5812, 5954, 6016, 6036]}

glv = GroupLossVariance(X, omega, G2, 1) #axis = 1 (0 rows e 1 columns)
RgrpIU = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NR):", RgrpIU)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)
print("RMSE: ", result)

##############################################################################################################################
algorithmImpartiality = AlgorithmImpartiality(X, omega, 1)
X_est = algorithmImpartiality.get_X_est(X_est) # calculates 1 estimated matrices

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [after the impartiality algorithm] -------------")

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i

# G group: identifying the groups (NR: number ratings)
# The configuration of groups was based in the hierarchical clustering (tree clustering - dendrogram)
# Clusters 1, 2 and 3
G2 = {1: [424, 549, 889, 1015, 1088, 1150, 1181, 1285, 1449, 1680, 1941, 1980, 2063, 2909, 3391, 3618, 3808, 3841, 4169, 4227, 4277, 4344, 4508, 4510, 5367, 5795, 5831], 2: [48, 53, 123, 148, 149, 173, 202, 216, 245, 302, 308, 319, 329, 331, 411, 438, 509, 528, 533, 543, 660, 692, 699, 721, 731, 770, 839, 855, 877, 984, 1050, 1068, 1069, 1112, 1117, 1120, 1125, 1137, 1194, 1203, 1207, 1224, 1242, 1246, 1264, 1266, 1274, 1298, 1303, 1317, 1333, 1340, 1354, 1422, 1425, 1451, 1465, 1470, 1496, 1579, 1613, 1632, 1635, 1671, 1675, 1737, 1741, 1748, 1749, 1764, 1780, 1812, 1835, 1837, 1884, 1889, 1897, 1899, 1920, 1926, 1943, 1988, 2010, 2012, 2030, 2073, 2077, 2092, 2109, 2124, 2453, 2544, 2793, 2857, 2878, 2887, 2934, 2962, 2986, 3018, 3029, 3118, 3182, 3280, 3285, 3308, 3312, 3320, 3336, 3389, 3410, 3462, 3475, 3476, 3483, 3491, 3519, 3562, 3589, 3610, 3648, 3650, 3681, 3683, 3693, 3705, 3724, 3768, 3821, 3823, 3834, 3842, 3884, 3885, 3929, 3934, 3942, 3999, 4016, 4021, 4033, 4048, 4054, 4083, 4089, 4140, 4186, 4305, 4345, 4354, 4387, 4411, 4482, 4578, 4579, 4682, 4728, 4732, 4802, 4867, 4957, 5011, 5015, 5054, 5074, 5107, 5111, 5220, 5306, 5312, 5387, 5433, 5493, 5501, 5504, 5511, 5530, 5536, 5550, 5605, 5675, 5682, 5747, 5759, 5763, 5878, 5880, 5888, 5916, 5996], 3: [195, 352, 482, 524, 531, 550, 678, 710, 752, 869, 881, 1010, 1019, 1051, 1383, 1447, 1448, 1605, 1647, 1676, 1698, 1733, 1880, 1912, 1958, 2015, 2106, 2116, 2181, 2304, 2507, 2529, 2665, 2777, 2820, 2907, 3032, 3067, 3163, 3224, 3272, 3292, 3311, 3401, 3471, 3507, 3526, 3539, 3626, 3675, 3778, 3792, 3824, 3829, 4041, 4064, 4085, 4238, 4312, 4386, 4425, 4447, 4448, 4543, 4647, 4725, 4808, 4979, 5026, 5046, 5100, 5256, 5333, 5394, 5614, 5627, 5636, 5643, 5788, 5812, 5954, 6016, 6036]}

glv = GroupLossVariance(X, omega, G2, 1) #axis = 1 (0 rows e 1 columns)
RgrpIU = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NR):", RgrpIU)

rmse = RMSE(X, omega)
result = rmse.evaluate(X_est)
print("RMSE: ", result)