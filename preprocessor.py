import os
import pandas as pd
import requests
import sys
from tqdm import tqdm
import time


def add_url(row):
      return f"https://www.imdb.com/title/tt{row}"


def add_rating(df):
  df_rating = pd.read_csv('data/ratings.csv')
  df_rating['movieId'] = df_rating['movieId'].astype(str)
  df_agg = df_rating.groupby('movieId').agg(
      rating_count = ('rating', 'count'),
      rating_avg = ('rating', 'mean')
  ).reset_index()

  df_rating_added = df.merge(df_agg, on='movieId')
  return df_rating_added


def add_poster(df):
  for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    tmdb_id = row['tmdbId']
    tmdb_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=aa0fc0f3fc9bb780b3d1f4d0316db6b4&language=en-US"
    result = requests.get(tmdb_url)
  
    try:
      df.at[i, "poster_path"] = "https://image.tmdb.org/t/p/original" + result.json()['poster_path']
      time.sleep(0.1)
    except (TypeError, KeyError) as e:
      df.at[i, 'poster_path'] = 'https://image.tmdb.org/t/p/original/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg'
  return df


if __name__ == "__main__":
  df_movie = pd.read_csv('data/movies.csv')
  df_movie['movieId'] = df_movie['movieId'].astype(str)
  df_link = pd.read_csv('data/links.csv', dtype=str)
  df_link['movieId'] = df_link['movieId'].astype(str)
  df_merged = df_movie.merge(df_link, on='movieId', how='left')
  df_merged['url'] = df_merged['imdbId'].apply(lambda x: add_url(x))
  df_result = add_rating(df_merged)
  df_result['poster_path'] = None
  df_result = add_poster(df_result)
  df_result.to_csv('data/movies_final.csv', index=None)
