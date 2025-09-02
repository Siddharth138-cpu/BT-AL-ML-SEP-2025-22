
# Titanic Survival Prediction Project


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Data
train = pd.read_csv(r"D:\projects\sid\archive (1)\titanic\train.csv")
test = pd.read_csv(r"D:\projects\sid\archive (1)\titanic\test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())

# Data Cleaning
# Fill missing Age with median
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

# Fill missing Fare with median
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Fill missing Embarked with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.countplot(x='Survived', data=train)
plt.title("Survival Distribution")

plt.subplot(1,3,2)
sns.barplot(x='Sex', y='Survived', data=train)
plt.title("Gender vs Survival")

plt.subplot(1,3,3)
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title("Pclass vs Survival")

plt.tight_layout()
plt.show()


#Feature Engineering
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

predictors = ['Pclass', 'IsFemale', 'Age', 'Fare'] + \
             [col for col in train.columns if col.startswith('Embarked_')]

X = train[predictors]
y = train['Survived']
X_test_final = test[predictors]


# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_val_scaled)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_val, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_log))
print("Classification Report:\n", classification_report(y_val, y_pred_log))

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_val)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_val, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_tree))
print("Classification Report:\n", classification_report(y_val, y_pred_tree))


final_predictions = log_reg.predict(X_test_scaled)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": final_predictions
})

submission.to_csv("titanic_submission.csv", index=False)
print("\n✅ Submission file 'titanic_submission.csv' created successfully!")
