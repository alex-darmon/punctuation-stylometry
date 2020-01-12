#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:52:29 2019

@author: alexandradarmon
"""

from pylatexenc.latexencode import unicode_to_latex
# 5.1. heatmaps (KL or other distance?)
import pandas as pd
import matplotlib.pyplot as plt
from punctuation.feature_operations.distances import d_KL
from punctuation.consistency.consistency_computation import (
        return_consistency_data,
        class_consistency
)

from punctuation.config import options
from punctuation.utils.utils import  load_corpus
from punctuation.recognition.training_testing_split import (
        get_nn_indexes
)
from punctuation.utils.utils import (
        try_to_load_as_pickled_object_or_None,
)

color1 = 'gist_earth'
color2 = 'gist_stern'

def get_comparison_df(feature_name_recognition,
                      nb_classes, index_train, index_test,
                      feature_name_consistency=options.feature_names[0],
                      col_name='author',df=None,
                      path_nn='data/pickle/recognition/author',
                      path_consistency='data/pickle/consistency/author',
                      path_comp='results/comparison/author',
                      col_unique='book_id',
                      populate_chunks_res=False,
                      distance=d_KL,chunk_size=75,
                      ):
    
    if df is None: df=load_corpus()
    
    df_compare = try_to_load_as_pickled_object_or_None('{}/neural_net_{}_{}.p'.format(path_nn,
                                              nb_classes, feature_name_recognition))
    
    df_compare['y_test'] = df[col_name].loc[index_test].tolist()
    df_compare[col_unique] = df[col_unique].loc[index_test].tolist()
    df_compare['is_not_correct'] = df_compare.apply(
            lambda row: int(row['predictions'] == row['y_test']), axis=1)
    
    df_reco = pd.merge(df[[col_unique, col_name]], df_compare, on=col_unique,
                       how='left')
    
    df_author_recognition = \
        df_reco.groupby(col_name, as_index=False)['is_not_correct'].mean()
    df_author_recognition['nb_docs'] =  \
        df.groupby(col_name)[col_unique].count().tolist()
    df_author_recognition =  pd.merge(df_author_recognition,
        df_compare.groupby('y_test',as_index=False)[col_unique].count().rename(
                columns={'y_test': col_name, col_unique:'test_size'}),
                on=col_name, how='left')
    
## CONSISTENCY
    
    df_consistency = return_consistency_data(feature_name_consistency, 
                                             distance=distance,
                                             path_consistency=path_consistency, 
                                             df=df,
                                             col=col_name, 
                                             col_unique=col_unique)
    
    
    df_author_consistency = class_consistency(df_consistency, feature_name_consistency,
                                            col_name=col_name)
    
## COMPARISON CONSISTENCY / ACCURACY


    df_class_comparison = pd.merge(df_author_consistency, df_author_recognition,
                                    on=col_name, how='inner')
    df_class_comparison.sort_values(
            '{}_compare'.format(feature_name_consistency), inplace=True)
    df_class_comparison.to_csv('{}/comparison_{}_{}_{}.csv'.format(path_comp, col_name,
                                feature_name_consistency, feature_name_recognition))
    
    if populate_chunks_res:
    
        df_class_comparison['{} (nb_documents - test size - consistency - accuracy)'.format(col_name)] = \
            df_class_comparison.apply(
                lambda row: '{} ({} - {} - {} - {})'.format(unicode_to_latex(row[col_name]),
                            row['nb_docs'],
                            row['test_size'],
                            round(row['{}_compare'.format(feature_name_consistency)], 3),
                            round(row['is_not_correct'], 3)), axis=1)
        
        for i in range(0, len(df_class_comparison), chunk_size*2):
            df_authori = df_class_comparison.iloc[i:i+chunk_size]
            df_authori2 = df_class_comparison.iloc[i+chunk_size:i+chunk_size*2]
            df_authori['{} (nb_documents - test size - consistency - accuracy) '.format(col_name)] = \
            df_authori2['{} (nb_documents - test size - consistency - accuracy)'.format(col_name)].tolist() + \
                                    [None] * max(0, i+chunk_size*2-len(df_class_comparison))
            
            df_authori[['{} (nb_documents - test size - consistency - accuracy)'.format(col_name),
                        '{} (nb_documents - test size - consistency - accuracy) '.format(col_name)]]\
                        .to_csv('{}/comparison_{}_{}_{}_{}.csv'.format( path_comp, col_name,
                                        feature_name_consistency,
                                        feature_name_recognition, i))
                        
    return (df_class_comparison, df_author_recognition,
            df_author_consistency)


##############################
### AVERAGE AND SHOW
def show_comparison(df_author_comparison, feature_name, bins=10,
                    path_comp='results/comparison/author'):
    
    index_bins = pd.qcut(df_author_comparison['{}_compare'.format(feature_name)], 
                                              bins)
    df_author_comparison['index_bins'] = index_bins
    
    df_bins = df_author_comparison.groupby('index_bins',as_index=False)\
            [['{}_compare'.format(feature_name), 'is_not_correct']].mean()
    
    
    plt.plot(range(bins), df_bins['{}_compare'.format(feature_name)],
             label='consistency', color='blue', marker='o', linestyle=':')
    plt.plot(range(bins), df_bins['is_not_correct'],
             label='recognition',  color='magenta', marker='o', linestyle=':')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #          fancybox=True, shadow=True)
    
    plt.savefig('{}/comparison_{}_{}_bins_{}.png'.format(path_comp,col_name,
                   feature_name,bins))
    
    plt.show()
    df_bins.to_csv('{}/comparison_{}_{}_bins_{}.csv'.format(path_comp,col_name,
                   feature_name,bins))

##############################

# AUTHOR 

feature_names = ['freq_pun', 'freq_nb_words', 'freq_length_sen',
                 'norm_transition_mat', '{f1,f3,f4,f5}', 'all']

df = load_corpus()
vector_size = [10, 50, 100, 200, 400, len(df.drop_duplicates('author'))]
col_name='author'
path_nn='data/pickle/recognition/author'


df_index_nn = get_nn_indexes(vector_size, df=df,
            col_class=col_name,
            path_index_nn =path_nn)

nb_classes = len(df.drop_duplicates('author'))# use 'all' features for the recognition.

(nb_classes,
 index_train, 
 index_test) = df_index_nn[df_index_nn['nb_classes']==nb_classes].iloc[0].to_list()
feature_name_recognition = feature_names[-1]
feature_name_consistency = options.feature_names[0]

(df_author_comparison, 
 df_author_recognition, 
 df_author_consistency) = get_comparison_df(feature_name_recognition,
                      nb_classes, index_train, index_test,
                      feature_name_consistency,
                      col_name='author',df=df,
                      path_nn='data/pickle/recognition/author',
                      path_consistency='data/pickle/consistency/author',
                      path_comp='results/comparison/author',
                      col_unique='book_id',
                      populate_chunks_res=True,
                      )


show_comparison(df_author_comparison, feature_name_consistency, bins=5,
                    path_comp='results/comparison/author')



##############################

# GENRE  

col_name = 'genre'
df_genre = load_corpus('data/pickle/genre_features.p')
vector_size = [10, 50, 100, 200, 400, len(df_genre.drop_duplicates(col_name))]
path_nn='data/pickle/recognition/genre'


df_index_nn = get_nn_indexes(vector_size, df=df_genre,
            col_class=col_name,
            path_index_nn =path_nn)

nb_classes = len(df_genre.drop_duplicates(col_name))

(nb_classes,
 index_train, 
 index_test) = df_index_nn[df_index_nn['nb_classes']==nb_classes].iloc[0].to_list()
feature_name_recognition = feature_names[-1]
feature_name_consistency = options.feature_names[0]

(df_genre_comparison, 
 df_genre_recognition, 
 df_genre_consistency) = get_comparison_df(feature_name_recognition,
                      nb_classes, index_train, index_test,
                      feature_name_consistency,
                      col_name='genre',df=df_genre,
                      path_nn='data/pickle/recognition/genre',
                      path_consistency='data/pickle/consistency/genre',
                      path_comp='results/comparison/genre',
                      col_unique='book_id',
                      populate_chunks_res=True,
                      chunk_size=16)

show_comparison(df_genre_comparison, feature_name_consistency, bins=5,
                    path_comp='results/comparison/genre')



##############################