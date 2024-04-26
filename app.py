import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(recipes_path, ratings_path):
    """Load and merge datasets."""
    try:
        recipes = pd.read_csv(recipes_path)
        ratings = pd.read_csv(ratings_path)
        data = pd.merge(recipes, ratings, on='id')
        logging.info("Data loaded and merged successfully.")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def preprocess_data(data, feature_columns):
    """Preprocess the data."""
    try:
        X = data[feature_columns]
        y = data['rating']
        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)
        logging.info("Data preprocessing completed.")
        return X, y
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
        raise

def train_model(X_train, y_train, n_estimators=100):
    """Train the RandomForestRegressor model."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model trained successfully.")
    return model

def main(args):
    data = load_data(args.recipes_path, args.ratings_path)
    X, y = preprocess_data(data, args.feature_columns.split(','))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, args.n_estimators)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse}")
    joblib.dump(model, os.path.join(args.model_dir, "recipe_rating_model.pkl"))
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to predict recipe ratings.")
    parser.add_argument('--recipes_path', type=str, required=True, help='Path to the recipes CSV file')
    parser.add_argument('--ratings_path', type=str, required=True, help='Path to the ratings CSV file')
    parser.add_argument('--feature_columns', type=str, required=True, help='Comma-separated list of feature columns')
    parser.add_argument('--model_dir', type=str, default='.', help='Directory to save the trained model')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the random forest')
    args = parser.parse_args()
    main(args)
