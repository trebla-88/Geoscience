import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data into a Pandas DataFrame
data = pd.read_csv('data/training_data.csv')

# Select the input variables and the target variable
X = data[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
y = data['Facies']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform feature engineering
X_train_squared = X_train ** 2
X_test_squared = X_test ** 2
X_train_interact = X_train[['GR', 'ILD_log10']] * X_train[['DeltaPHI', 'PHIND']]
X_test_interact = X_test[['GR', 'ILD_log10']] * X_test[['DeltaPHI', 'PHIND']]
X_train_new = pd.concat([X_train, X_train_squared, X_train_interact], axis=1)
X_test_new = pd.concat([X_test, X_test_squared, X_test_interact], axis=1)

# Perform LDA to select the most discriminative features
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
X_train_lda = lda.transform(X_train_scaled)
X_test_lda = lda.transform(X_test_scaled)

# Train a random forest classifier with hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_lda, y_train)
model = grid.best_estimator_

# Evaluate the performance of the model on the test set
y_pred = model.predict(X_test_lda)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))