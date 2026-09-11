import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Load Dataset
data = pd.read_csv("heartdiseaseuci.csv")

print("Dataset shape:", data.shape)
print(f"The total null values in data {data.isnull().sum().sum()} \n")
missing_values = data.isnull().sum()
total_cell = np.prod(data.shape)
print(
    f"The percentage of missing values: {(missing_values.sum() / total_cell) * 100:.2f} % \n"
)
print(data.describe())
print(data.info())
print(f"\nFirst 5 rows:\n{data.head(5)}\n")

# 2. Data Cleaning & Preprocessing
data = data.drop_duplicates()

# Drop 'id' column if present (it is an index identifier, not a predictive feature)
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Dynamically locate the target column ('num' in UCI dataset)
target_col = [
    col
    for col in data.columns
    if col.lower() in ["target", "output", "num", "condition", "heartdisease"]
][0]

# Convert multiclass target 'num' (0, 1, 2, 3, 4) into binary (0 = No Disease, 1 = Disease Present)
data[target_col] = (data[target_col] > 0).astype(int)

# Fill Missing Values (NaN Imputation)
# Numerical columns: Fill NaN with column median
num_cols = data.select_dtypes(include=["int64", "float64"]).columns.drop(
    target_col, errors="ignore"
)
for col in num_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# Categorical columns: Fill NaN with column mode (most frequent item)
cat_cols = data.select_dtypes(
    include=["object", "category", "string", "str"]
).columns
for col in cat_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])

# Separate features X and target y
X = data.drop(target_col, axis=1).copy()
y = data[target_col].copy()

# Encode string/categorical features into numeric values
categorical_cols = X.select_dtypes(
    include=["object", "category", "string", "str"]
).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}\n")

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training
# Model 1: Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# Model 2: Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_prob = dt_model.predict_proba(X_test_scaled)[:, 1]


# 6. Model Evaluation
def print_metrics(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== {model_name} Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")


print_metrics("Logistic Regression", y_test, lr_pred)
print_metrics("Decision Tree", y_test, dt_pred)

# 7. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)
disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr, display_labels=["No Disease", "Disease"]
)
disp_lr.plot(ax=axes[0, 0], cmap="Blues", colorbar=False)
axes[0, 0].set_title("Logistic Regression Confusion Matrix")

# Plot 2: Decision Tree Confusion Matrix
cm_dt = confusion_matrix(y_test, dt_pred)
disp_dt = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt, display_labels=["No Disease", "Disease"]
)
disp_dt.plot(ax=axes[0, 1], cmap="Greens", colorbar=False)
axes[0, 1].set_title("Decision Tree Confusion Matrix")

# Plot 3: ROC Curve Comparison
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
roc_auc_dt = auc(fpr_dt, tpr_dt)

axes[1, 0].plot(
    fpr_lr,
    tpr_lr,
    color="blue",
    lw=2,
    label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})",
)
axes[1, 0].plot(
    fpr_dt,
    tpr_dt,
    color="green",
    lw=2,
    label=f"Decision Tree (AUC = {roc_auc_dt:.2f})",
)
axes[1, 0].plot([0, 1], [0, 1], color="navy", linestyle="--")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].set_title("ROC Curve Comparison")
axes[1, 0].legend(loc="lower right")

# Plot 4: Decision Tree Feature Importance
importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values()
importances.plot(kind="barh", ax=axes[1, 1], color="teal")
axes[1, 1].set_title("Decision Tree Feature Importance")
axes[1, 1].set_xlabel("Importance Score")

plt.tight_layout()
plt.show()