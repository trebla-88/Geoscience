import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data into a pandas DataFrame
df = pd.read_csv('data/training_data.csv')
df = df.reset_index()
#print(df.sort_values(by='Facies', ascending=True))

df['DeltaPHI'] = [np.exp(delta) for delta in df['DeltaPHI']]

gb = df.groupby('Facies')
f_df = [gb.get_group(k) for k in gb.groups]

def density_by_var(var):
    plt.figure()
    for df in f_df:
        sns.kdeplot(data=df, x=var, fill=True, cmap="Blues")
    plt.title("Distribution de la densité en fonction du " + var)
    plt.xlabel(var)
    plt.legend([str(k) for k in range(1,10)])
    plt.savefig("figures/" + var + ".png")

for var in ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]:
    density_by_var(var)