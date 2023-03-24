import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('data/PCA_transformed_data.csv')
# Split the data into features and target
X = data.iloc[:, 1:4].values
y = data.iloc[:, 0].values.astype(int)


# Train the MLPClassifier on the data
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42,)
clf.fit(X, y)


# Load the new data to predict
new_data = pd.read_csv('data/PCA_transformed_well_data.csv')

# Make sure the new data has the same length as the training data
#new_data = new_data.iloc[:, :X.shape[1]]

# Make predictions on the new data
predictions = clf.predict(new_data.iloc[:,1:4])
#print(confusion_matrix(new_data.iloc[:,0], predictions))
print("PCA accuracy: " + str(accuracy_score(new_data.iloc[:,0], predictions)))