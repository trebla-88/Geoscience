import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Charger les données dans un DataFrame Pandas
df = pd.read_csv('data/training_data.csv')

#Neural network
X = df.iloc[:,4:9].values
y = df.iloc[:,0].values.astype(int)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs')

#Import well_data
df_well = pd.read_csv('data/well_data_with_facies.csv')

# Sélectionner les variables d'entrée et la variable cible
X = df_well[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
y = df_well['Facies']

#Prediction des données
y_pred = clf.predict(df_well[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']])

print(confusion_matrix(y, y_pred))
print(accuracy_score(y, y_pred))