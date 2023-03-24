import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/PCA_transformed_data.csv')

# Define the variables to plot
vars_to_plot = list(data.columns)

# Remove the target variable from the list
vars_to_plot.remove('Facies')

# Loop over the variables and create a plot for each one
for var in vars_to_plot:
    # Split the data into 9 subsets based on the target variable
    f_df = [data[data['Facies'] == i] for i in range(1, 10)]

    # Create a new figure
    plt.figure()

    # Loop over the subsets and create a KDE plot for each one
    for df in f_df:
        sns.kdeplot(data=df, x=var, fill=True, cmap="Blues")

    # Set the title and axis labels
    plt.title("Distribution de la densité en fonction du " + var)
    plt.xlabel(var)

    # Set the legend labels
    legend_labels = [str(k) for k in range(1, 10)]

    # Add the legend
    plt.legend(legend_labels)

    # Save the figure
    plt.savefig("figures/PCA/" + var + ".png")