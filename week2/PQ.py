"""Practice Questions:
1. Train a logistic regression model on Titanic dataset.
2. Create confusion matrix and accuracy score.
3. Compare DecisionTreeClassifier and KNeighborsClassifier accuracy.
4. Tune max_depth of Decision Tree and note differenc"""

import pandas as pd
import matplotlib
import seaborn as sns
pd.options.display.max_rows = 999
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

 
data = pd.read_csv("/media/madhur/New Volume/BESKILLED/WEEK1/MINIPROJECT1/Titanic_Cleaned.csv")
print(data.info())
data = data[
    ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
]
print(data.head())

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,    # 20 for testing
    random_state=42, #The data is randomly shuffled before splitting.
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Actual:", y_test.values)
print("Predicted:", y_pred)
accuracy = accuracy_score(y_test, y_pred)


cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(8,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")

plt.show()

tree = DecisionTreeClassifier(
    random_state=42
)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
print("Decision Tree Accuracy:", tree_accuracy)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("KNN Accuracy:", knn_accuracy)
print("Logistic Regression:", accuracy)
print("Decision Tree:", tree_accuracy)

for depth in [1, 2, 3, 4, 5, 6, 8, 10, None]:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    pred = tree.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("max_depth:", depth, "Accuracy:", acc)