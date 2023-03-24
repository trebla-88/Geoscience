import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# Charger les données dans un DataFrame Pandas
df = pd.read_csv('data/LDA_transformed_data.csv')

#Neural network
from sklearn.neural_network import MLPClassifier
X = df.iloc[:,1:].values
y = df.iloc[:,0].values.astype(int)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs')

#Import well_data
df_well = pd.read_csv('data/well_data_with_facies.csv')

#LDA of well_data
# Sélectionner les variables d'entrée et la variable cible
X = df_well[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
y = df_well['Facies']

# Effectuer une AFD
lda = LinearDiscriminantAnalysis()
n=lda.fit(X, y)
ldafact_well=lda.transform(X)

y_pred = clf.predict(ldafact_well)

print(confusion_matrix(y, y_pred))