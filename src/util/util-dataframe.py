# python 3.x
import pandas as pd
# List of Tuples
fruit_list = [  ]
#Create a DataFrame object
df = pd.DataFrame(fruit_list, columns = ['UserID', 'Name' , 'Price', 'Stock'])

df.set_index('UserID', inplace = True)

#Add new ROW
df.loc[1]=[ 'Mango', 4, 'No' ]
df.loc[2]=[ 'Apple', 14, 'Yes' ]
print(df)