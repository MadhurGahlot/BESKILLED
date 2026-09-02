# Mini Project 2: House Price Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/BESKILLED/week2/Housing.csv")

print("Dataset shape:", data.shape)
#print(data.head())
print(f"The total null values in data {data.isnull().sum().sum()} \n")
missing_values = data.isnull().sum()
total_cell = np.prod(data.shape)
print(f"The percentage of the missing values {(missing_values/total_cell)*100} % \n")
print(data.describe())
print(data.info())
print(f" \n {data.head(5)} ")

label_columns = [
"mainroad",
"guestroom",
"basement",
"hotwaterheating",
"airconditioning",
"prefarea"
]

numeric_columns = [
"area",
"bedrooms",
"bathrooms",
"stories",
"parking"
]

LabelEncoder = LabelEncoder()
for columns in label_columns:
    data[columns] = LabelEncoder.fit_transform(data[columns])

#one hot coding 
data = pd.get_dummies(
data,
columns=["furnishingstatus"],
drop_first=True,
dtype=int
)

print(data.describe())
print(f'data info : {data.info()}')

X = data.drop("price",axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nFirst 10 predicted prices:")
print(y_pred[:10])

print("\nFirst 10 actual prices:")
print(y_test.iloc[:10].values)

r2 = r2_score(y_test, y_pred)

print("\nR² Score:", r2)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("RMSE:", rmse)


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show() 