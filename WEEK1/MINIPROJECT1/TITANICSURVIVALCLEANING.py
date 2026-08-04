# Mini Project 1: “Titanic Survival Prediction – Data Cleaning Project”
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("D:/BESKILLED/WEEK1/Titanic-Dataset.csv")
print(data.info())
print(data.describe())
print(f'THE NULL VALUE IN DATASET IS \n {data.isnull()}')
print(f'THE  MISSING VALUE IN EACH COLUMN \n {data.isnull().sum()}')
print(f'Total Missing Values In The Entire DataSet: \n {data.isnull().sum().sum()}')

#CLEANING THE DATA 

data_cleaned = data.drop(columns=["Cabin"])  # this column have the 687 missing values out of 891
print(data_cleaned)

# FILL THE DATA : 
mode_value = data_cleaned["Embarked"].mode()[0]
data_cleaned["Embarked"] = data_cleaned["Embarked"].fillna(mode_value)

#print(F'THE  COLUMN EMBARKED AFTER FILLING \n {data_cleaned.loc[:50,"Embarked"]}')
print(f'The Missing value after the filling Embarked \n {data_cleaned.isnull().sum()}')

#Fill the age with the mean

data_cleaned["Age"] = data_cleaned["Age"].fillna(data_cleaned["Age"].mean())
print(F'THE DATASET AFTER CLEANING \n {data_cleaned.isnull().sum()} ')

# HERE WE WILL ENCODE THE SEX AND THE EMBARKED WITH LABEL ENCODING 

le = LabelEncoder()

data_cleaned["Sex"] = le.fit_transform(data_cleaned["Sex"])

#print(f'THE DATASET AFETR THE SEX ENCODING  \n : {data_cleaned.loc[:50,"Sex"]}')

data_cleaned["Embarked"] = le.fit_transform(data_cleaned["Embarked"])

print(f'THE DATASET AFETR THE ENCODING: \n {data_cleaned.loc[:50,["Sex","Embarked"]]}')
# print(f'THE DATASET : {data_cleaned.to_string()}') # print the entire data

plt.hist(data_cleaned["Age"],bins=20)
plt.title("THE AGE DISTRIBUTION ")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# SAVE THE DATA
data_cleaned.to_csv("Titanic_Cleaned.csv", index=False)

