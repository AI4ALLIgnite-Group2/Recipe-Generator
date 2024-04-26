# -*- coding: utf-8 -*-
"""Edit Dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1egPuJSQ0PBN_jFZcQW8KyDIh1lqNrPa7

# **Importing Libraries and Data Set**
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import ast
import joblib
import matplotlib.pyplot as plt

"""# **Load the datasets**

"""

ingredient = pd.read_pickle("/content/drive/MyDrive/Colab Notebooks/Data/ingr_map.pkl")
recipes = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Data/PP_recipes.csv")
names = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Data/RAW_recipes.csv")
ratings = pd.read_csv("/content/drive/MyDrive/RAW_interactions.csv")
ingr_map = pd.read_pickle("/content/drive/MyDrive/Colab Notebooks/Data/ingr_map.pkl")

"""# **Display the first few entries of the recipes DataFrame to verify it's loaded correctly**

"""

print(recipes.head())

"""# Display the first few entries of the ingredient DataFrame to check the *details*"""

print(ingredient.head())

"""# **Checking the size of the datasets to understand their dimensions**"""

print("Names dataset: Entries =", len(names), ", Columns =", len(names.columns))
print("Recipes dataset: Entries =", len(recipes), ", Columns =", len(recipes.columns))

"""# Merging the recipes and names datasets on the 'id' column to combine details"""

new_df = recipes.merge(names, left_on="id", right_on="id")
print("New merged DataFrame size: Entries =", len(new_df), ", Columns =", len(new_df.columns))

"""# Merging ratings data to include average ratings for each recipe

"""

new_df = new_df.merge(ratings.groupby('recipe_id')['rating'].mean(), left_on='id', right_index=True, how='left')
print("Data after merging with ratings: ", new_df.columns)

"""# **Querying and Filtering Data**

# Example queries for specific ingredients
"""

print(ingredient[ingredient['replaced']=="cheese"].head())

print(ingredient[ingredient['replaced']=="brown rice"].head())

print(ingredient[ingredient['replaced']=="avocado"].head())

"""# Filtering recipes that include specific ingredient IDs


"""

two_ids = [790, 255]
mask1 = new_df['ingredient_ids'].apply(lambda x: ' 790' in x)
df1 = new_df[mask1]
mask2 = new_df['ingredient_ids'].apply(lambda x: ' 1170' in x)
df2 = new_df[mask2]

"""# Combining the filtered dataframes to find recipes that include both ingredients

"""

combined = df1.merge(df2, left_on="id", right_on="id")
print("Combined DataFrame for two ingredients:", len(combined))

"""# **Sample Data Display**"""

random_recipe = combined.sample(1)
print("Selected Recipe Name:", list(random_recipe['name_x'])[0])
print("------------------\n")

print("Ingredients:\n")
for item in eval(list(random_recipe['ingredients_y'])[0]):
    print("- " + item)
print("------------------\n")
print("Recipe Steps:\n")
for step in eval(list(random_recipe['steps_x'])[0]):
    print("- " + step)

"""##Test Accuracy##

## Data Preprocessing ##
"""

for col in new_df.columns:
    if pd.api.types.is_numeric_dtype(new_df[col]):
        new_df[col] = pd.arrays.SparseArray(new_df[col])

# Check the conversion
print(new_df.dtypes)

"""# Ensure all sparse columns have a fill compatibility with COO matrix conversion


"""

for col in new_df.columns:
    if pd.api.types.is_numeric_dtype(new_df[col]):
        # Ensure the fill value is set to 0 for compatibility with COO matrix conversion
        new_df[col] = pd.arrays.SparseArray(new_df[col], fill_value=0)

"""
#Convert the sparse columns into a COO matrix
"""

from scipy.sparse import csr_matrix

# Select only the sparse columns
sparse_columns = new_df.select_dtypes(include='Sparse')

# Attempt to convert to COO then to CSR format
sparse_matrix = csr_matrix(sparse_columns.sparse.to_coo())

# Now your sparse_matrix is ready for use in algorithms that support sparse input

"""# Check the fill value of a sparse column


"""

# Check the fill value of a sparse column
print(new_df['calorie_level'].sparse.fill_value)

# Select only the sparse columns
sparse_columns = new_df.select_dtypes(include='Sparse')

# Convert to COO matrix and then to CSR format
sparse_matrix = csr_matrix(sparse_columns.sparse.to_coo())

# Your sparse_matrix is now in CSR format, ready for use

"""## Prepare the Target Variable ##

"""

y = new_df['rating'].copy()

"""# Fill missing values in 'y' to avoid NaN issues during modeling

"""

if y.isnull().any():
    y = y.fillna(y.mean())  # One way to handle missing values - fill them with the mean of the column

"""## Model Training ##

"""

# Define the target variable from the DataFrame
y = new_df['rating'].copy()

# Handle missing values if necessary
if y.isnull().any():
    y = y.fillna(y.mean())

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sparse_matrix, y, test_size=0.2, random_state=42)

# Training a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

"""# Calculate and display feature importances for model interpretation

"""

feature_importances = model.feature_importances_
# names aligned with your matrix columns
features = [f"Feature {i}" for i in range(sparse_matrix.shape[1])]
for importance, name in sorted(zip(feature_importances, features), reverse=True)[:10]:  # top 10 features
    print(f"{name}: {importance}")

"""# Plot residuals to visually inspect model performance and check for patterns

"""

residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.xlabel('Actual Ratings')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()