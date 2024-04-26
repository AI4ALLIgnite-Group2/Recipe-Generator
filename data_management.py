import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess data by handling missing values and encoding categorical variables.
    Args:
    data (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Fill missing numerical data with the median
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].median(), inplace=True)
    
    # Encode categorical variables using dummy variables
    data = pd.get_dummies(data, drop_first=True)
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
    return pd.merge(recipes, ratings, on=key)

def main():
    # Define paths to the datasets
    recipes_path = '/path/to/recipes.csv'
    ratings_path = '/path/to/ratings.csv'

    # Load the data
    recipes = load_data(recipes_path)
    ratings = load_data(ratings_path)

    # Preprocess the datasets
    recipes = preprocess_data(recipes)
    ratings = preprocess_data(ratings)

    # Merge the datasets
    data = merge_datasets(recipes, ratings)

    # Further processing or save to file, etc.
    print(data.head())

if __name__ == '__main__':
    main()
