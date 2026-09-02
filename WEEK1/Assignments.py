'''1.Load a dataset using Pandas and summarize basic stats '''
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np
 # pd.options.display.max_rows = 999 # to print the entire rows
data = pd.read_csv("Titanic-Dataset.csv")

print(data)
data_info = pd.read_csv("Titanic-Dataset.csv").info()
#print(data_info)
data_describe = pd.read_csv("Titanic-Dataset.csv").describe()
#print(data_describe)

# check the missing values

print(f'THE MISSING VALUES : \n {data.isnull()}')
print(f'THE TOTAL MISSING VALUES: \n {data.isnull().sum()}')
print(f'TOTAL MISSING VALUE IN WHOLE DATA SETS: \n {data.isnull().sum().sum()}')
missing = data.isnull().sum().sum()
total_cells = np.prod(data.shape)
missing_percentage = (missing/total_cells)*100
print(f'The percentage of missing value in dataset {missing_percentage}')

#check the values column by column

#print(data.iloc[50:100,11])

data_cleaned = pd.read_csv("Titanic-Dataset.csv")

mode_value = data_cleaned["Embarked"].mode()[0]
data_cleaned["Embarked"] = data_cleaned["Embarked"].fillna(mode_value)

print(f'After Fill The value of Embarked: \n {data_cleaned.isnull().sum()}')

# now fill the age using mean
data_cleaned["Age"] = data_cleaned["Age"].fillna(data_cleaned["Age"].mean())
print(f'The data after filling  the vakue of age: \n {data_cleaned.isnull().sum()}')

# the missing  value of cabin is too high so we drop this table 
data_cleaned=data_cleaned.drop(columns=["Cabin"])
print(f'THE DATA AFTER THE DROPING CABIN COLUMN: \n {data_cleaned.isnull().sum()}')

#Now thw data is completly cleaned

le = LabelEncoder()

data_cleaned["Sex"] = le.fit_transform(data_cleaned["Sex"])
#print(data_cleaned.loc[:50,"Sex"])

encoded_Embarked = pd.get_dummies(data_cleaned, columns=["Embarked"])

print(encoded_Embarked)




