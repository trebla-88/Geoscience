import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Charger les données dans un DataFrame Pandas
data = pd.read_csv('data/training_data.csv')

# Sélectionner les variables d'entrée et la variable cible
X = data[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
y = data['Facies']

# Effectuer une AFD
lda = LinearDiscriminantAnalysis()
n=lda.fit(X, y)
ldafact=lda.transform(X)
print(ldafact)



# Transformer les données en utilisant l'AFD
X_lda = lda.transform(X)