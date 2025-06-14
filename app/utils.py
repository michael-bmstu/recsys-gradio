import pandas as pd
import numpy as np
import dvc.api
from pathlib import Path
from scipy.special import erfc
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle
from typing import Any, List, Callable, Optional
import os

params = dvc.api.params_show()
SEED = params['seed']
DATA = Path(params['data_root'])
WEIGHTS = Path(params['weight_root'])
np.random.seed(SEED)

def chauvenet(array):
    n = array.shape[0]
    m = array.mean()
    s_p = array.std()
    mask = erfc(abs(array - m)/s_p) < 0.5 / n
    return mask

def flatmap(f: Callable[[Any], List[Any]], items: List[Any] | np.ndarray) -> chain[Any]:
    return chain.from_iterable(map(f, items))

def genre_splitter(genre_names: str) -> list[str]:
    return genre_names.split(", ")

def get_mean_genre(df: pd.DataFrame, x: pd.Series) -> Optional[float]:
    s = x.sum()
    if s: return (x * df.loc[x.index, 'rating']).sum() / s
    else: return None

def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    anime = pd.read_csv(DATA / 'anime.csv', index_col='anime_id')
    anime.dropna(inplace=True)

    ratings = pd.read_csv(DATA / 'rating.csv')
    ratings = ratings[ratings['rating'] >= 0]

    return anime, ratings

def prepare_data(is_save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    anime, ratings = read_data()

    count_reviews = ratings.groupby('user_id').size()
    outlier_users = count_reviews[chauvenet(count_reviews.values)]
    bad_user_threshold = outlier_users.min()
    count_reviews_df = pd.DataFrame({'user_id': count_reviews.index, 'count_reviews': count_reviews.values})
    ratings = pd.merge(ratings, count_reviews_df, on='user_id', how='inner')
    ratings = ratings[ratings['count_reviews'] < bad_user_threshold]
    median_cnt = ratings['count_reviews'].median()
    ratings = ratings[ratings['count_reviews'] >=  median_cnt]
    ratings = ratings.drop(columns=['count_reviews'])

    outlier_anime = anime[chauvenet(anime['members'])]
    anime.drop(outlier_anime.index, inplace=True)
    outlier_anime = anime[chauvenet(anime['rating'])]
    anime.drop(outlier_anime.index, inplace=True)

    if is_save:
        anime.to_csv(DATA / 'anime_clean.csv')
        ratings.to_csv(DATA / 'rating_clean.csv')

    return anime, ratings

def agg_anime(is_save: bool = True) -> pd.DataFrame:
    anime, ratings = prepare_data()
    m_uniq = anime['genre'].unique()
    genres = set(flatmap(genre_splitter, m_uniq))
    genres = list(genres) # всего 43 жанра

    anime['genres'] = anime['genre'].apply(lambda g: g.split(', '))
    one_hot_df = anime.explode('genres')
    one_hot_encoded = pd.get_dummies(one_hot_df['genres'])
    anime_genre = one_hot_encoded.groupby('anime_id').sum()

    df_agg = pd.merge(ratings, anime_genre, on='anime_id', how='inner')
    df_grouped = df_agg.groupby('user_id')[genres].agg(lambda x: get_mean_genre(df_agg, x))
    mean_rating = df_agg[genres].mean()
    # df_grouped[genres].fillna(mean_rating, inplace=True)
    df_grouped[genres] = df_grouped[genres].fillna(mean_rating)

    if is_save:
        df_grouped.to_csv(DATA / 'anime_train.csv')
        anime_genre.to_csv(DATA / 'anime_genre.csv')
    return df_grouped

def fit_model(df_train: pd.DataFrame, is_save: bool = True):
    scaler = MinMaxScaler()
    df_gr_scaled = scaler.fit_transform(df_train)
    clustering = KMeans(n_clusters=4, random_state=SEED)
    kmeans = clustering.fit(df_gr_scaled)

    if is_save:
        with open(WEIGHTS / 'kmeans.pkl', mode='wb') as f:
            pickle.dump(kmeans, f)
        with open(WEIGHTS / 'scaler.pkl', mode='wb') as f:
            pickle.dump(scaler, f)
    return kmeans, scaler

def first_run():
    if not os.path.exists(WEIGHTS):
        os.makedirs(WEIGHTS)
    fit_model(agg_anime())

def read_prepare():
    with open(WEIGHTS / 'kmeans.pkl', mode='rb') as f:
        kmeans = pickle.load(f)
    with open(WEIGHTS / 'scaler.pkl', mode='rb') as f:
        scaler = pickle.load(f)

    anime_clean = pd.read_csv(DATA / 'anime_clean.csv', index_col='anime_id')
    anime_genre = pd.read_csv(DATA / 'anime_genre.csv', index_col='anime_id')
    return anime_clean, anime_genre, kmeans, scaler

def get_best_anime(cluster: np.ndarray, anime: pd.DataFrame, genres: List[str], top_k: int = 10, combo:int = 4):
    top_genres_ind = np.argsort(cluster)[-top_k:]
    top_genres = np.array(genres)[top_genres_ind]
    combo_anime = np.array([0] * anime.shape[0])
    for g in top_genres:
        combo_anime += anime[g]
    result_anime = anime[combo_anime >= combo]
    if result_anime.shape[0]:
        return result_anime['anime_id'].unique()
    return get_best_anime(cluster, anime, genres, top_k + 1, combo - 1)

def recomendate(ratings: np.ndarray, hist: np.ndarray, n_recs: int = 5):
    with open(WEIGHTS / 'kmeans.pkl', mode='rb') as f:
        kmeans = pickle.load(f)
    with open(WEIGHTS / 'scaler.pkl', mode='rb') as f:
        scaler = pickle.load(f)

    anime = pd.read_csv(DATA / 'anime_clean.csv')
    anime_genre = pd.read_csv(DATA / 'anime_genre.csv')

    genres = list(anime_genre.columns)
    ratings = scaler.transform(ratings)
    label = kmeans.predict(ratings)[0]
    cluster = kmeans.cluster_centers_[label]
