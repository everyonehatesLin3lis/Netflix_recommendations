import os
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify

app = Flask(__name__)

base_dir = os.path.join(os.getcwd())

model_file = os.path.join(base_dir, 'Model', 'lightgbm_user_movie_recommendation.txt')
parquet_file = os.path.join(base_dir, 'Data', 'Data_for_recommendations.parquet')
csv_file = os.path.join(base_dir, 'Data', 'movie_titles.csv')

model = lgb.Booster(model_file=model_file)

df = pd.read_parquet(parquet_file)

movie_titles = pd.read_csv(csv_file)

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

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Recommend movies for a given user.

    Returns:
        str: JSON representation of the recommended movies.
    """
    user_id = int(request.args.get('user_id'))
    recommendations = generate_recommendations(user_id, df, model, movie_titles)
    if not recommendations.empty:
        return recommendations.to_json(orient='records')
    else:
        return jsonify({"message": f"No recommendations generated for user ID {user_id}"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)