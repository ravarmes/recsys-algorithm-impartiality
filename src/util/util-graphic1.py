import matplotlib.pyplot as plt 
import numpy as np 
  
  
x = np.array([3, 5, 10, 15]) 
y = np.array([0.000153749, 0.000150642, 0.000148635, 0.000143388])
  
plt.plot(x, y) 
plt.xlabel("k")  
plt.ylabel("Rgrp")  
plt.title("Results of the Group Injustice Rgrp for the MovieLens dataset")  
plt.show() 