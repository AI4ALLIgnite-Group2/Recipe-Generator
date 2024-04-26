import pandas as pd
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(ingredient_path, recipes_path, names_path, ratings_path):
    """
    Load datasets based on the provided file paths.
    """
    try:
        ingredient = pd.read_pickle(ingredient_path)
        recipes = pd.read_csv(recipes_path)
        names = pd.read_csv(names_path)
        ratings = pd.read_csv(ratings_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)
    return ingredient, recipes, names, ratings

def merge_data(recipes, names, ratings):
    """
    Merge recipes, names, and ratings dataframes on the 'id' column.
    """
    try:
        new_df = recipes.merge(names, on='id')
        new_df = new_df.merge(ratings.groupby('recipe_id')['rating'].mean(), left_on='id', right_index=True, how='left')
        logging.info("Data merged successfully.")
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        sys.exit(1)
    return new_df

def preprocess_data(dataframe):
    """
    Process dataframe to prepare for modeling. Includes handling sparse data.
    """
    try:
        for col in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                dataframe[col] = pd.arrays.SparseArray(dataframe[col], fill_value=0)
        logging.info("Data preprocessed successfully.")
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        sys.exit(1)
    return dataframe

def train_model(X_train, y_train, n_estimators=100):
    """
    Train RandomForestRegressor on the provided training data.
    """
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
    except Exception as e:
        logging.error(f"Error training model: {e}")
        sys.exit(1)
    return model

def main(args):
    ingredient, recipes, names, ratings = load_data(args.ingredient_path, args.recipes_path, args.names_path, args.ratings_path)
    new_df = merge_data(recipes, names, ratings)
    new_df = preprocess_data(new_df)
    y = new_df['rating'].fillna(new_df['rating'].mean()).copy()
    X = new_df.drop(columns=['rating'])  # Adjust based on actual feature columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, args.n_estimators)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--ingredient_path', type=str, required=True, help='Path to the ingredient data file')
    parser.add_argument('--recipes_path', type=str, required=True, help='Path to the recipes data file')
    parser.add_argument('--names_path', type=str, required=True, help='Path to the names data file')
    parser.add_argument('--ratings_path', type=str, required=True, help='Path to the ratings data file')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the random forest')
    args = parser.parse_args()
    main(args)
