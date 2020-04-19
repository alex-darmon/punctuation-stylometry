#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:57:26 2019

@author: alexandradarmon
"""

import pandas as pd
from punctuation.recognition.training_testing_split import (
        get_nn_indexes
)
from punctuation.feature_operations.distances import d_KL
from punctuation.recognition.recognition_algorithms import (
        launch_nearest_neighbour,
        launch_neural_net
)

from punctuation.config import options
from punctuation.utils.utils import (
        load_corpus,
)


### Neural Network features RECOGNITION
list_features = [options.freq_pun_col,
                 options.freq_nb_words_col,
                 options.freq_length_sen_with_col,
                 options.norm_transition_mat_col,
                 options.freq_pun_col+\
                 options.freq_nb_words_col\
                 +options.freq_length_sen_with_col\
                 +options.norm_transition_mat_col,
                 options.freq_pun_col+\
                 options.freq_nb_words_col+\
                 options.freq_length_sen_with_col+\
                 options.transition_mat_col+ options.norm_transition_mat_col+options.mat_nb_words_pun_col,
                 ]
feature_names = ['freq_pun', 'freq_nb_words', 'freq_length_sen',
                 'norm_transition_mat', '{f1,f3,f4,f5}', 'all']



### AUTHOR RECOGNITION
df = load_corpus()
vector_size = [10, 50, 100, 200, 400, len(df.drop_duplicates('author'))]

df_index_nn = get_nn_indexes(vector_size, df,
                col_class='author',
                test_size=0.2, random_state=8,
                path_index_nn = 'data/pickle/recognition/author')

launch_nearest_neighbour(df, df_index_nn.iloc[:-1],
                         list_features=options.feature_names,
                         col_class='author',
                         path_index_nn = 'data/pickle/recognition/author')


launch_neural_net(df, df_index_nn,list_features=list_features,
                      feature_names=feature_names,
                      col_class='author',
                      path_index_nn = 'data/pickle/recognition/author',
                      path_res='results/recognition/author')



### GENRE RECOGNITION
df_genre = load_corpus('data/pickle/genre_features.p')
vector_size = [len(df_genre.drop_duplicates('genre'))]


df_index_nn_genre = get_nn_indexes(vector_size, df_genre,
                col_class='genre',
                test_size=0.2, random_state=8,
                path_index_nn = 'data/pickle/recognition/genre')


#launch_nearest_neighbour(df_genre, df_index_nn_genre,
#                         list_features=options.feature_names,
#                         col_class='genre',
#                         path_index_nn='data/pickle/recognition/genre')


launch_neural_net(df_genre, df_index_nn, list_features=list_features,
                  feature_names=feature_names,
                  col_class='genre',
                  path_index_nn = 'data/pickle/recognition/genre',
                  path_res='results/recognition/genre')


