import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Load the data into a Pandas DataFrame
data = pd.read_csv('data/training_data.csv')
well=pd.read_csv('data/well_data_with_facies.csv')
databis=data.iloc[:,0]
wellbis=well.iloc[:,-1]

# Select the variables of interest
X = data[['GR', 'ILD_log10', 'PE', 'DeltaPHI', 'PHIND']]
X_well=well[['GR', 'ILD_log10', 'PE', 'DeltaPHI', 'PHIND']]

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(X)
well_scaled=scaler.fit_transform(X_well)
# Create a PCA object with 2 components
pca = PCA(n_components =3)

# Fit the PCA object to the data
pca.fit(X_Scaled)


# Transform the data into the new PCA space
X_pca = pca.transform(X)
well_pca=pca.transform(X_well)
sns.pairplot(pd.DataFrame(X_pca))
plt.show()
# Print the explained variance ratio for each component
#print(pca.explained_variance_ratio_)

# Print the transformed data
#print(X_pca)

#print(X_pca.shape)
new_data=pd.concat([databis,pd.DataFrame(X_pca)],axis=1)
well_data=pd.concat([wellbis,pd.DataFrame(well_pca)],axis=1)
well_data.to_csv('data/PCA_transformed_well_data.csv',index=False)
new_data.to_csv('data/PCA_transformed_data.csv',index=False)
#print(new_data)

# Create a PCA object with all components
#pca = PCA()

# Fit the PCA object to the data
#pca.fit(X)

# Calculate the cumulative explained variance ratio
#cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)

# Print the cumulative explained variance ratio
#print(cumulative_var_ratio)