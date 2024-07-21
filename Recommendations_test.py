import os
import pandas as pd
import lightgbm as lgb
import random

base_dir = os.path.dirname(os.path.abspath(__file__))

model_file = os.path.join(base_dir, 'Model', 'lightgbm_user_movie_recommendation.txt')
parquet_file = os.path.join(base_dir, 'Data', 'Data_for_recommendations.parquet')
csv_file = os.path.join(base_dir, 'Data', 'movie_titles.csv')

print("Base directory:", base_dir)
print("Model file path:", model_file)
print("Parquet file path:", parquet_file)
print("CSV file path:", csv_file)

try:
    model = lgb.Booster(model_file=model_file)
except FileNotFoundError:
    print(f"Model file not found: {model_file}")
    print("Please check the path and ensure the file exists.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

try:
    df = pd.read_parquet(parquet_file)
except FileNotFoundError:
    print(f"Parquet file not found: {parquet_file}")
    print("Please check the path and ensure the file exists.")
except pd.errors.EmptyDataError as e:
    print(f"No data: {e}")
except Exception as e:
    print(f"An error occurred while loading the parquet file: {e}")

try:
    movie_titles = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"CSV file not found: {csv_file}")
    print("Please check the path and ensure the file exists.")
except UnicodeDecodeError as e:
    print(f"Encoding error: {e}")
except pd.errors.ParserError as e:
    print(f"Parser error: {e}")
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")

def generate_recommendations(user_id: int, df: pd.DataFrame, model: lgb.Booster, movie_titles: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Generate movie recommendations for a user.

    Args:
        user_id (int): The ID of the user.
        df (pd.DataFrame): DataFrame containing all necessary data.
        model (lgb.Booster): Trained LightGBM model.
        movie_titles (pd.DataFrame): DataFrame containing movie titles.
        top_n (int, optional): Number of top recommendations to return. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame containing the top N recommended movies for the user.
    """

    if user_id not in df['customerID'].values:
        print(f"User ID {user_id} not found in the dataset.")
        return pd.DataFrame()

    user_data = df[df['customerID'] == user_id].iloc[0]

    movie_data = df.drop_duplicates(subset=['movieID']).copy()
    user_movie_data = movie_data[movie_data['customerID'] != user_id]

    user_movie_data['user_genre_rating'] = user_data['user_genre_rating']
    user_movie_data['user_year_rating'] = user_data['user_year_rating']
    user_movie_data['user_avg_rating'] = user_data['user_avg_rating']
    user_movie_data['user_num_ratings'] = user_data['user_num_ratings']

    features = [
        'user_genre_rating', 'user_year_rating', 'movie_avg_rating', 
        'user_avg_rating', 'genres', 'year', 'user_num_ratings', 
        'movie_num_ratings', 'popularity', 'runtime', 'original_language'
    ]

    X = user_movie_data[features]
    user_movie_data['predicted_rating'] = model.predict(X)

    recommendations = user_movie_data.sort_values(by='predicted_rating', ascending=False).head(top_n)

    recommendations = recommendations.merge(movie_titles, on='movieID', how='left')
    
    return recommendations[['movieID', 'title', 'predicted_rating']]

#Test

customer_movie_counts = df.groupby('customerID')['movieID'].nunique()

customers_watched_more_than_15 = customer_movie_counts[customer_movie_counts > 20].index.tolist()

random_customerID = random.choice(customers_watched_more_than_15)
print('UserID: ', random_customerID)

user_id = random_customerID
recommendations = generate_recommendations(user_id, df, model, movie_titles)
if not recommendations.empty:
    print(recommendations)
else:
    print(f"No recommendations generated for user ID {user_id}")