import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

data = pd.read_csv('2016-ml-contest/training_data.csv')

data.describe()

test_well = data[data['Well Name'] == 'SHANKLE']
data = data[data['Well Name'] != 'SHANKLE']

features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND','PE','NM_M', 'RELPOS']
feature_vectors = data[features]
facies_labels = data['Facies']

sns.pairplot(feature_vectors[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND','PE']])

scaler = StandardScaler().fit(feature_vectors)
scaled_features = scaler.transform(feature_vectors)

X_train, X_cv, y_train, y_cv = \
train_test_split(scaled_features, facies_labels,
test_size=0.05, random_state=42)

clf = svm.SVC(C=10, gamma=1)
clf.fit(X_train, y_train)

y_test = test_well['Facies']

well_features = test_well.drop(['Facies',
'Formation',
'Well Name',
'Depth'],
axis=1)

X_test = scaler.transform(well_features)

y_pred = clf.predict(X_test)
test_well['Prediction'] = y_pred

target_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D','PS', 'BS']

print(classification_report(y_test, y_pred, target_names=target_names))