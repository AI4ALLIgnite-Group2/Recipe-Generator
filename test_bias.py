import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Or RandomForestClassifier for categorical ratings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

# Function to load datasets
def load_datasets(ingredient_path, recipes_path, names_path, ratings_path):
    ingredient = pd.read_pickle(ingredient_path)
    recipes = pd.read_csv(recipes_path)
    names = pd.read_csv(names_path)
    ratings = pd.read_csv(ratings_path)
    return ingredient, recipes, names, ratings

# Function to convert IDs to names using a mapping table
def convert_ids_to_names(recipes, ingr_map):
    id_to_ingredient = pd.Series(ingr_map['replaced'].values, index=ingr_map['id']).to_dict()
    recipes['ingredient_names'] = recipes['ingredient_tokens'].apply(lambda ids: [id_to_ingredient.get(id, "Unknown") for id in ids])
    return recipes

# Function to preprocess and merge features
def preprocess_and_merge(recipes):
    mlb = MultiLabelBinarizer()
    ingredients_encoded = mlb.fit_transform(recipes['ingredient_names'])
    ingredients_df = pd.DataFrame(ingredients_encoded, columns=mlb.classes_)
    full_features = pd.concat([recipes.drop(['ingredient_tokens', 'ingredient_names'], axis=1), ingredients_df], axis=1)
    return full_features

# Main function to run the analysis
def main():
    ingredient_path = "data/ingr_map.pkl"
    recipes_path = "data/PP_recipes.csv"
    names_path = "data/RAW_recipes.csv"
    ratings_path = "data/RAW_interactions.csv"
    
    ingredient, recipes, names, ratings = load_datasets(ingredient_path, recipes_path, names_path, ratings_path)
    recipes = convert_ids_to_names(recipes, ingredient)
    full_features = preprocess_and_merge(recipes)

    # Example analysis: Distribution of calorie levels
    sns.histplot(data=recipes, x='calorie_level', bins=30)
    plt.title('Distribution of Calorie Levels')
    plt.show()

if __name__ == "__main__":
    main()
