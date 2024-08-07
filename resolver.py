import pandas as pd

item_fname = 'data/movies_final.csv'

def random_items():
    df_movie = pd.read_csv(item_fname)
    df_movie = df_movie.fillna('')
    result_items = df_movie.sample(n=10).to_dict("records")
    return result_items

def random_genres_items(genre):
    df_movie = pd.read_csv(item_fname)
    df_genre = df_movie[df_movie['genres'].apply(lambda x: genre in x.lower())]
    df_genre = df_genre.fillna('')
    result_items = df_genre.sample(n=10).to_dict("records")
    return result_items