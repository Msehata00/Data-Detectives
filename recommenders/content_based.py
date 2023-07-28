"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
   # Convert the movie genres into a single string for each movie
    #movies['genres'] = movies['genres'].str.replace('|', ' ')

    # Combine genres to create movie descriptions
    movies['description'] = movies['genres']

    # Convert the movie descriptions into a Tfidf matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['description'])

    # Get the indices of the selected favorite movies
    idx_movies = movies[movies['title'].isin(movie_list)].index.tolist()

    if not idx_movies:
        raise ValueError("No movies from movie_list found in the dataset. Please check the movie titles.")

    # Calculate similarity scores between the selected favorite movies and all other movies
    similarity_scores = cosine_similarity(tfidf_matrix[idx_movies], tfidf_matrix)

    # Get the average similarity scores for each movie across the selected favorite movies
    avg_similarity_scores = similarity_scores.mean(axis=0)

    # Get the indices of the top_n recommended movies based on the average similarity scores
    top_n_indices = avg_similarity_scores.argsort()[::-1][:top_n]

    # Get the titles and genres of the top_n recommended movies
    recommended_movies = movies.iloc[top_n_indices][['title', 'genres']].values.tolist()
    recommended_ratings = []
    for movie_title, _ in recommended_movies:
        movie_ratings = ratings[ratings['movieId'] == movies[movies['title'] == movie_title]['movieId'].values[0]]['rating']
        if not movie_ratings.empty:
            # Calculate the average rating for the recommended movie
            average_rating = movie_ratings.mean()
            recommended_ratings.append(average_rating)
        else:
            recommended_ratings.append(None)
    recommended_movies_with_ratings = [(movie_title, genres, rating) for (movie_title, genres), rating in zip(recommended_movies, recommended_ratings)]

    return recommended_movies_with_ratings

movie_list = []
# Assuming you have a test set with known ratings (actual_ratings) and a model (content_model)
# Get the recommended movies and predicted ratings from the model
try:
    recommended_movies, recommended_ratings = content_model(movie_list, top_n=10)
    print("Recommended Movies with Ratings:")
    for movie_title, genres, rating in recommended_movies:
        print(f"Movie: {movie_title}, Genres: {genres}, Predicted Rating: {rating}")
except ValueError as e:
    print(f"Error: {e}")


def hyperparameter_tuning(movie_data):
    # Define the hyperparameter grid for the TfidfVectorizer
    param_grid = {
        'max_features': [1000, 2000, 3000],  # The maximum number of features (vocabulary size)
        'min_df': [1, 2, 3],  # The minimum number of documents a word must be present in to be kept
        'max_df': [0.7, 0.8, 0.9],  # The maximum percentage of documents a word can be present in to be kept
    }

    # Initialize the TfidfVectorizer and GridSearchCV
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    grid_search = GridSearchCV(tfidf_vectorizer, param_grid, cv=5)

    # Fit the GridSearchCV on the movie data to find the best hyperparameters
    grid_search.fit(movie_data['description'])

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    return best_params