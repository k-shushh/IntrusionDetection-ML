import pandas as pd
import numpy as np

#Data Preprocessing
df = pd.read_csv("synthetic_network_intrusion.csv")

print(df.isnull().sum())  
df.fillna(0, inplace=True)  # Replace missing values with 0

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Attack Type'] = encoder.fit_transform(df['Attack Type']) #Attack Type values turned into numeric values

df = pd.get_dummies(df, columns=['Protocol']) #Protocol values turned into numeric values

#print(df.dtypes)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')  # Ensure all numeric
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  # Scale


#Training and Testing data
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Attack Type'])
y = df['Attack Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Introduction
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Initialize models
rf_model = RandomForestRegressor(n_estimators= 50,random_state=42)
svm_model = SVC(kernel='rbf', gamma = 1.0)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)  
y_test = encoder.transform(y_test)

# Train models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predictions
rf_pred = rf_model.predict(X_test)
rf_pred = np.round(rf_pred).astype(int)
svm_pred = svm_model.predict(X_test)

# Print Evaluation Metrics
print("Random Forest Performance:\n", classification_report(y_test, rf_pred))
print("SVM Performance:\n", classification_report(y_test, svm_pred))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.show()

import numpy as np

features = X.columns
importances = rf_model.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(8, 5))
plt.title("Feature Importance")
plt.bar(range(len(indices[:10])), importances[indices[:10]], align="center")
plt.xticks(range(len(indices[:10])), [features[i] for i in indices[:10]], rotation=45)

plt.show()


