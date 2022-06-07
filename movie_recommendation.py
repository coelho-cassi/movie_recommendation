import pandas as pd
import numpy as np

import sklearn
from sklearn.decomposition import TruncatedSVD

from flask import Flask, jsonify

app = Flask = Flask(__name__)

columns = ['user_id', 'movie_id','rating', 'timestamp']
user_data = pd.read_csv('ml-100k/u.data', names=columns, sep ='\t');

movie_data = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1');

columns = ['movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_data = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=columns)

merged_data = pd.merge(user_data, movie_data, on='movie_id')

data_pivot_table = merged_data.pivot_table(index='user_id', values ='rating', columns='movie title', fill_value=0)

transposed_data = data_pivot_table.values.T

SVD = TruncatedSVD(n_components=12)
resultant_matrix = SVD.fit_transform(transposed_data)

correlation_matrix = np.corrcoef(resultant_matrix)

movie_names = data_pivot_table.columns
movie_list = list(movie_names)


@app.route('/<string:movie>', methods=['GET'])
def get_related_movies(movie):
    if movie in movie_list:
        movie_index = movie_list.index(movie)
        movie_corr = correlation_matrix[movie_index]
        related_movies = list(movie_names[(movie_corr > 0.85) & (movie_corr < 1)])
        return jsonify(related_movies), 200
    else:
        return "Error: Movie not on list", 404


if __name__ == '__main__':
    app.run(debug= True)
