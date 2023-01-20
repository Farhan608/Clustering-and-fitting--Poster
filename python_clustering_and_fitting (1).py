# -*- coding: utf-8 -*-
"""Python Clustering and Fitting.ipynb


"""

from sklearn.cluster import KMeans

# Load the data into a Pandas dataframe
import pandas as pd
df = pd.read_csv("/Data.csv")
df = df.drop(columns=['2020 [YR2020]', '2021 [YR2021]'])
# List of columns with the '..' value
columns = ['1990 [YR1990]', '2000 [YR2000]', '2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']

# Iterate over the columns
for column in columns:
    # Drop rows with the '..' value in the current column
    df = df[df[column] != '..']
df.drop(columns=['1990 [YR1990]', '2000 [YR2000]'], inplace=True)
df.head()

"""In this code, we are loading a data set containing CO2 emissions for various countries into a Pandas dataframe. Then, we are dropping the columns '2020 [YR2020]' and '2021 [YR2021]', as the csv file contains missing data for these columns for every record.

Next, we are creating a list of columns with the value '..', which indicates missing data. We are then iterating over this list of columns for years and dropping any rows that have the value '..' in the current column. This step is necessary to ensure that we are working with complete and accurate data.

Finally, we are dropping the columns '1990 [YR1990]' and '2000 [YR2000]', as they are not needed for our analysis. This leaves us with a dataframe containing only the relevant data.
"""

# Select the columns to cluster
X = df[['2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run the K-Means algorithm
kmeans = KMeans(n_clusters=15)
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the dataframe
df['cluster'] = labels

# Group the data by cluster and country
grouped = df.groupby(['cluster', 'Country Name'])

# View the cluster results
print(grouped)

"""The code above is performing the following steps:

1. Selecting the columns to cluster: The relevant columns are selected and stored in the X variable. These columns contain the CO2 emissions values for the years 2012 to 2019.
2. Scaling the data: The data is scaled using the StandardScaler method from scikit-learn. Scaling the data is important because the columns have different scales, which can affect the performance of the clustering algorithm.
3. Running the K-Means algorithm: The KMeans algorithm from scikit-learn is initialized with n_clusters=15 and then fit to the scaled data.
4. Getting the cluster labels: The cluster labels for each row in the data are obtained using the labels_ attribute of the kmeans object.
5. Adding the cluster labels to the dataframe: The cluster labels are added as a new column to the dataframe.
6. Grouping the data by cluster and country: The data is grouped by both the cluster label and the country name. The resulting object is a Pandas DataFrameGroupBy object, which allows us to perform various operations on the grouped data.

You can then use this grouped data to create a pie chart or other visualizations to explore the clustering results. For example, you could use the grouped.size() method to get the size (i.e., number of rows) of each group, and then use this information to create a pie chart showing the distribution of countries across the different clusters.
"""

for cluster, group in grouped:
    print(f'Cluster {cluster}: {group["Country Name"].tolist()}')

"""This code appears to be iterating through a grouped dataframe, with the variable 'grouped' representing the dataframe and 'cluster' and 'group' representing the index and corresponding rows, respectively.
For each iteration, the code is printing the cluster number and a list of the "Country Name" values for the rows in the group.
"""

# Group the data by cluster label
grouped = df.groupby('cluster')

# Initialize lists for the values and labels of the pie chart
values = []
labels = []

# Iterate over the grouped data
for cluster, group in grouped:
    # Append the size of the group to the values list
    values.append(group.shape[0])
    # Append the cluster label and the total number of countries in the cluster to the labels list
    labels.append(f'Cluster {cluster} ({group.shape[0]})')

# Set the colors of the pie chart
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

plt.figure(figsize=(15, 15))
# Create the pie chart
plt.pie(values, labels=labels, colors=colors)

# Set the font size of the labels
plt.rcParams['font.size'] = 10

# Create a legend for the pie chart
plt.legend(title='Countries')

# Display the pie chart
plt.show()

import matplotlib.pyplot as plt

# Get the cluster sizes
sizes = grouped.size()

# Get the cluster labels and country counts
clusters = sizes.index.get_level_values(0)
counts = sizes.values

# Create the bar chart
plt.bar(clusters, counts)
plt.xlabel('Cluster')
plt.ylabel('Number of countries')
plt.title('Countries per cluster')
plt.show()

# Continue from the previous code block
import matplotlib.pyplot as plt

# Select the columns to plot
X = df[['2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run the KMeans algorithm
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_

# Get the cluster centers
centers = kmeans.cluster_centers_

# Create the scatter plot
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='+', c='k', s=200)
plt.xlabel('2012-2019 CO2 emissions')
plt.ylabel('Other years CO2 emissions')
plt.title('Cluster membership and cluster centers')
plt.show()

df.columns

import numpy as np
from scipy.optimize import curve_fit

# Select the data to fit
y = df['2014 [YR2014]'].values
x = np.arange(len(y))

# Define the exponential growth model
def exp_growth(x, a, b):
    return a * np.exp(b * x)

# Fit the model to the data
params, cov = curve_fit(exp_growth, x, y)
a, b = params

# Calculate the lower and upper limits of the confidence range
lower, upper = err_ranges(exp_growth, x, y, params, cov, alpha=0.95)

# Make predictions for the next 10 years
x_pred = np.arange(max(x) + 1, max(x) + 11)
y_pred = exp_growth(x_pred, *params)

# Calculate the lower and upper limits of the confidence range for the predictions
lower_pred, upper_pred = err_ranges(exp_growth, x_pred, y_pred, params, cov, alpha=0.95)

# Print the predictions and the confidence range
print(f'Predicted population for the next 10 years: {y_pred}')
print(f'Confidence range: ({lower_pred}, {upper_pred})')