import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

# Create a MinMaxScaler object
minMaxScaler = MinMaxScaler()
# Create a StandardScaler object
standardScaler = StandardScaler()

# Read the CSV file into a DataFrame using the read_csv() function
data = pd.read_csv("housing.csv")

longitude = data['longitude']
latitude = data['latitude']

# Use the select_dtypes() method to select only the columns
# with numeric data (i.e. the arithmetic subset)
arithmetic_subset = data.select_dtypes(include=['int64', 'float64'])


# Use the select_dtypes() method to select only the columns
# with non-numeric data (i.e. the categorical subset)
categorical_subset = data.select_dtypes(exclude=['int64', 'float64'])


# Use the fit_transform() method to apply standardization
# to the columns of the DataFrame
standard_scaled_data = pd.DataFrame(standardScaler.fit_transform(arithmetic_subset), columns=arithmetic_subset.columns)

# Get the median values of each column by using the median() method 
# to fill the missing values 
median_values = standard_scaled_data.median()

# Fill the missing values of the dataframe with the median values of each column
# using the fillna() method and the median_values variable that we calculated earlier
standard_scaled_data = standard_scaled_data.fillna(median_values)
print(standard_scaled_data)

# Check if there are any remaining missing values
if standard_scaled_data.isnull().any().any():
    print("There are missing values in the data.")
else:
    print("There are no missing values in the data.")


# Use the Pandas get_dummies() function to create one-hot vectors
one_hot_vectors = pd.get_dummies(categorical_subset['ocean_proximity'])

# Add the one-hot vectors to the original DataFrame
categorical_subset = pd.concat([categorical_subset, one_hot_vectors], axis=1)
print(categorical_subset)

pdfs = {}
for column in standard_scaled_data.columns:
    pdf = scipy.stats.gaussian_kde(standard_scaled_data[column])
    pdfs[column] = pdf

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
for ax, column, pdf in zip(axs.flat, standard_scaled_data.columns, pdfs.values()):
    standard_scaled_data[column].hist(bins=50, ax=ax, density=True)
    x = np.linspace(standard_scaled_data[column].min(), standard_scaled_data[column].max())
    y = pdf.evaluate(x)
    ax.plot(x, y)
    ax.set_title(column)
plt.show()




