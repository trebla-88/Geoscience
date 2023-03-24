import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données dans un DataFrame Pandas
data = pd.read_csv('data/training_data.csv')
databis=data.iloc[:,0]

# Sélectionner les variables d'entrée et la variable cible
X = data[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']]
y = data['Facies']

# Effectuer une AFD
lda = LinearDiscriminantAnalysis()
n=lda.fit(X, y)
ldafact=lda.transform(X)

new_data=pd.concat([databis,pd.DataFrame(ldafact)],axis=1)
new_data.columns = ['Facies', 'LDA_0', 'LDA_1', 'LDA_2', 'LDA_3', 'LDA_4']

gb = new_data.groupby('Facies')
f_df = [gb.get_group(k) for k in gb.groups]

def density_by_var(var):
    plt.figure()
    for df in f_df:
        sns.kdeplot(data=df, x=var, fill=True, cmap="Blues")
    plt.title("Distribution de la densité en fonction du coeff LDA " + var)
    plt.xlabel(var)
    plt.legend([str(k) for k in range(1,10)])
    plt.savefig("figures/LDA/" + var + ".png")

for k in range(5):
    density_by_var('LDA_' + str(k))

# Transformer les données en utilisant l'AFD
#X_lda = lda.transform(X)
#print(X_lda)