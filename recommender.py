import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import pickle
saved_model_fname = "model/finalized_model.sav"
data_fname = "data/ratings.csv"
item_fname = "data/movies_final.csv"
weight = 10

def model_train():
    df_rating = pd.read_csv(data_fname)
    df_rating["userId"] = df_rating["userId"].astype("category")
    df_rating["movieId"] = df_rating["movieId"].astype("category")

    # Sparse matrix
    rating_matrix = coo_matrix(
        (
            df_rating["rating"].astype(np.float32),
            (
                df_rating["movieId"].cat.codes.copy(),
                df_rating["userId"].cat.codes.copy(),
            ),
        )
    )
    
    als_model = AlternatingLeastSquares(
        factors=50, regularization=0.01, dtype=np.float64, iterations=50
    )
    
    als_model.fit(weight * rating_matrix)
    
    pickle.dump(als_model, open(saved_model_fname, "wb"))
    return als_model


def calculate_item_based(item_id, items):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recs = loaded_model.similar_items(itemid=int(item_id), N=11)
    return [str(items[r]) for r in recs[0]]


def item_based_recommendation(item_id):
    df_rating = pd.read_csv(data_fname)
    df_rating["userId"] = df_rating["userId"].astype("category")
    df_rating["movieId"] = df_rating["movieId"].astype("category")
    df_movie = pd.read_csv(item_fname)
    
    items = dict(enumerate(df_rating["movieId"].cat.categories))
    try:
        parsed_id = df_rating["movieId"].cat.categories.get_loc(int(item_id))
        result = calculate_item_based(parsed_id, items)
    except KeyError as e:
        result = []
    result = [int(x) for x in result if x != item_id]
    result_items = df_movie[df_movie["movieId"].isin(result)].to_dict("records")
    return result_items


def calculate_user_based(user_items, items):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recs = loaded_model.recommend(
        userid=0, user_items=user_items, recalculate_user=True, N=10
    )
    return [str(items[r]) for r in recs[0]]


def build_matrix_input(input_rating_dict, items):
    model = pickle.load(open(saved_model_fname, "rb"))
    
    item_ids = {r: i for i, r in items.items()}
    mapped_idx = [item_ids[s] for s in input_rating_dict.keys() if s in item_ids]
    data = [weight * float(x) for x in input_rating_dict.values()]
    rows = [0 for _ in mapped_idx]
    shape = (1, model.item_factors.shape[0])
    return coo_matrix((data, (rows, mapped_idx)), shape=shape).tocsr()


def user_based_recommendation(input_ratings):
    df_rating = pd.read_csv(data_fname)
    df_rating["userId"] = df_rating["userId"].astype("category")
    df_rating["movieId"] = df_rating["movieId"].astype("category")
    df_movie = pd.read_csv(item_fname)
    
    items = dict(enumerate(df_rating["movieId"].cat.categories))
    input_matrix = build_matrix_input(input_ratings, items)
    result = calculate_user_based(input_matrix, items)
    result = [int(x) for x in result]
    result_items = df_movie[df_movie["movieId"].isin(result)].to_dict("records")
    return result_items    


if __name__ == "__main__":
    model = model_train()
