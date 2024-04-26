import pandas as pd
import sys
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data.

    Raises:
    FileNotFoundError: If the file cannot be found at the specified path.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while loading data from {file_path}: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess data by handling missing values and encoding categorical variables.
    Args:
    data (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Fill missing numerical data with the median
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            logging.info(f"Filled missing values in {col} with median value {median_val}")

    # Encode categorical variables using dummy variables
    data = pd.get_dummies(data, drop_first=True)
    logging.info("Categorical variables encoded")
    return data

def merge_datasets(recipes, ratings, key='id'):
    """
    Merge two datasets on a common key.
    Args:
    recipes (pd.DataFrame): The recipes DataFrame.
    ratings (pd.DataFrame): The ratings DataFrame.
    key (str): The key on which to merge the datasets.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    try:
        merged_data = pd.merge(recipes, ratings, on=key)
        logging.info("Datasets merged successfully")
        return merged_data
    except pd.errors.MergeError as e:
        logging.error(f"Failed to merge datasets: {e}")
        sys.exit(1)

def main():
    # Define paths to the datasets
    recipes_path = 'data/recipes.csv'
    ratings_path = 'data/ratings.csv'

    # Load the data
    recipes = load_data(recipes_path)
    ratings = load_data(ratings_path)

    # Preprocess the datasets
    recipes = preprocess_data(recipes)
    ratings = preprocess_data(ratings)

    # Merge the datasets
    data = merge_datasets(recipes, ratings)

    # Example of further processing or save to file, etc.
    logging.info(f"Processed data: \n{data.head()}")

if __name__ == '__main__':
    main()
