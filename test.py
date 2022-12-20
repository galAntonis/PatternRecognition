# Importing libraries
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import math

# Read the CSV file into a DataFrame using the read_csv() function
data = pd.read_csv("housing.csv")

# Create a MinMaxScaler object
minMaxScaler = MinMaxScaler()

# Use the select_dtypes() method to select only the columns
# with numeric data (i.e. the arithmetic subset)
arithmetic_subset = data.select_dtypes(include=['int64', 'float64'])
# Use the select_dtypes() method to select only the columns
# with non-numeric data (i.e. the categorical subset)
categorical_subset = data.select_dtypes(exclude=['int64', 'float64'])
# Use the Pandas get_dummies() function to create one-hot vectors
categorical_subset = pd.get_dummies(categorical_subset['ocean_proximity'])

# Use the fit_transform() method to apply min-max scaling
# to the columns of the DataFrame
minmax_scaled_data = pd.DataFrame(minMaxScaler.fit_transform(arithmetic_subset), columns=arithmetic_subset.columns)


# Get the median values of each column by using the median() method 
# to fill the missing values 
median_values = minmax_scaled_data.median()
# Fill the missing values of the dataframe with the median values of each column
# using the fillna() method and the median_values variable that we calculated earlier
minmax_scaled_data = minmax_scaled_data.fillna(median_values)

pdfs = {}
for column in minmax_scaled_data.columns:
    pdf = gaussian_kde(minmax_scaled_data[column])
    pdfs[column] = pdf
for column in categorical_subset.columns:
    pdf = gaussian_kde(categorical_subset[column])
    pdfs[column] = pdf
fig, axs = plt.subplots(2, 5, sharex='col', sharey='row')
axs[0, 0].hist(minmax_scaled_data["longitude"], bins=50)
axs[0, 1].hist(minmax_scaled_data["latitude"], bins=50)
axs[0, 2].hist(minmax_scaled_data["housing_median_age"],bins=50)
axs[0, 3].hist(minmax_scaled_data["total_rooms"], bins=50)
axs[0, 4].hist(minmax_scaled_data["total_bedrooms"], bins=50)
axs[1, 0].hist(minmax_scaled_data["population"],bins=50)
axs[1, 1].hist(minmax_scaled_data["households"],bins=50)
axs[1, 2].hist(minmax_scaled_data["median_income"],bins=50)
axs[1, 3].hist(minmax_scaled_data["median_house_value"],bins=50)
axs[1, 4].hist(data["ocean_proximity"],bins=4)

plt.show()

