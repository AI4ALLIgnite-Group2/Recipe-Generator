import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Set a random seed for reproducibility
seed = 42

# Paths to the dataset files
recipes_path = "/path/to/recipes.csv"
ratings_path = "/path/to/ratings.csv"

# Load your datasets
recipes = pd.read_csv(recipes_path)
ratings = pd.read_csv(ratings_path)

# Merging recipes and ratings data on 'id', which is a common column in both datasets
data = pd.merge(recipes, ratings, on='id')

# Example preprocessing: Selecting features and target variable
# Replace 'feature1', 'feature2', 'feature3' with actual feature column names from your dataset
X = data[['feature1', 'feature2', 'feature3']]
y = data['rating']  # Target variable for prediction

# Handling missing values, if there are any
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

# Initializing and training the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=seed)
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Calculating the Mean Squared Error (MSE) for the predictions
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model to a file for later use
joblib.dump(model, "recipe_rating_model.pkl")
