'''Build Linear Regression model on housing dataset (predict price)'''
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
matplotlib.use("TkAgg") # this for plt.show in mint

data_housing =  pd.read_csv("Housing.csv")

print(data_housing.head(10))

data_info =  data_housing.info()
print(data_info)

data_describe = data_housing.describe()
print(data_describe)

data_null_values = data_housing.isnull().sum()
print(data_null_values)
print("THERE IS NO NULL VALUE IN THIS DATA SET")
missing_value = data_null_values.sum()
total_cells = np.prod(data_housing.shape)
missing_percentage =(missing_value/total_cells)*100
print(f'The percentage of missing value: {missing_percentage} %')


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(data_housing["area"],
    data_housing["price"]/1000000,c=data_housing['price']/1000000,cmap='viridis')
plt.xlabel("Area")
plt.ylabel("Price (in millions)")
plt.colorbar()
plt.title("Area vs House Price")


plt.subplot(2, 2, 2)
parking_count = data_housing["parking"].value_counts().sort_index()
plt.bar(parking_count.index, parking_count.values)
plt.xlabel("No of Parking")
plt.ylabel("Number of Houses")
plt.title("Parking Data")


plt.subplot(2,2,3)
plt.hist(data_housing["price"] / 1000000)
plt.xlabel("Price (In millions)")
plt.ylabel("No. of House")
plt.show()

print(data_housing.loc[:10,["parking","price","area"]])

data = pd.get_dummies(data_housing, drop_first=True) # convert the value into the 0 1 

# Separate input features (X) and target (y)
X = data.drop("price", axis=1)
y = data["price"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

print(y_pred)