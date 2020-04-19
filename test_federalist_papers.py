#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:42:11 2020

@author: alexandradarmon
"""

import pandas as pd
from punctuation.config import options
from punctuation.parser.punctuation_parser import (
 enrich_features
 )
papers = {
    'Madison': [10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    'Hamilton': [1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 24,
                 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 59, 60,
                 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                 78, 79, 80, 81, 82, 83, 84, 85],
    'Jay': [2, 3, 4, 5],
    'Shared': [18, 19, 20],
    'Disputed': [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63],
    'TestCase': [64]
}


def read_files_into_string(filenames):
    strings = []
    for filename in filenames:
        with open(f'data/data_federalist/federalist_{filename}.txt') as f:
            strings.append(f.read())
    return '\n'.join(strings)



federalist_by_author = {}
for author, files in papers.items():
    federalist_by_author[author] = read_files_into_string(files)


authors = ("Hamilton", "Madison", "Disputed", "Jay", "Shared")

federalist = pd.DataFrame(data=None)
list_authors = []
list_bookid = []
list_texts = []
for author, files in papers.items():
    list_authors += [author]*len(files)
    list_bookid += files
    for filename in files:
        with open(f'data/data_federalist/federalist_{filename}.txt') as f:
            list_texts.append(f.read())
            
federalist['author'] = list_authors
federalist['book_id'] = list_bookid
federalist['text'] = list_texts
enrich_features(federalist, text_col='text')





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

df_index_nn = get_nn_indexes([len(federalist.drop_duplicates('author'))], federalist,
                col_class='author',
                test_size=0.2, random_state=8,
                path_index_nn = '', save=False)


res_nearest_neighbour = launch_nearest_neighbour(federalist, df_index_nn,
                         list_features=options.feature_names,
                         col_class='author',
                         path_index_nn = '')


launch_neural_net(federalist, df_index_nn,
                  list_features=list_features,
                      feature_names=feature_names,
                      col_class='author',
                      path_index_nn = '',
                      path_res='')

