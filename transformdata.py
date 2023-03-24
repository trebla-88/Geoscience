import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the data into a Pandas DataFrame
data = pd.read_csv('data/well_data_with_facies.csv')
databis=data.iloc[:,-1]
print(databis)

# Select the variables of interest
X = data[['GR', 'ILD_log10', 'PE', 'DeltaPHI', 'PHIND']]

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(X)
# Create a PCA object with 2 components
pca = PCA(n_components =3)

# Fit the PCA object to the data
pca.fit(X_Scaled)

# Transform the data into the new PCA space
X_pca = pca.transform(X)


# Print the explained variance ratio for each component
#print(pca.explained_variance_ratio_)

# Print the transformed data
#print(X_pca)

#print(X_pca.shape)
new_data=pd.concat([databis,pd.DataFrame(X_pca)],axis=1)
print(new_data)

# Create a PCA object with all components
#pca = PCA()

# Fit the PCA object to the data
#pca.fit(X)

# Calculate the cumulative explained variance ratio
#cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)

# Print the cumulative explained variance ratio
#print(cumulative_var_ratio)