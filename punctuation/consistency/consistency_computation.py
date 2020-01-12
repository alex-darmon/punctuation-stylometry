#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:01:49 2019

@author: alexandradarmon
"""

import pandas as pd
import scipy as sp
import numpy as np
from punctuation.feature_operations.distances import d_KL
from punctuation.utils.utils import (
        try_to_load_as_pickled_object_or_None, 
        load_corpus,
        save_as_pickled_object)
import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn-white')


def merge_df(df, feature,
             col_to_merge='author',
             col_unique='book_id',
             suffix_left='', suffix_right='_y'):
    
    df_merged = pd.merge(df[[col_to_merge,feature,'book_id']],
                     df[[col_to_merge,feature,col_unique]],
                     on=col_to_merge, suffixes=(suffix_left,suffix_right))

    df_merged = df_merged[df_merged\
                ['{}{}'.format(col_unique,
                                suffix_left)]!=\
                df_merged['{}{}'.format(col_unique, suffix_right)]]

    return df_merged


def get_distance_column(df_merged, feature,  col='author', distance=d_KL,
                 suffix_left='', suffix_right='_y', col_name=None):
    if col_name is None:
        col_name = feature+'_compare'
    feature_x = feature+suffix_left
    feature_y = feature+suffix_right
    df_merged[col_name] = df_merged.apply(lambda row: \
             distance(row[feature_x], row[feature_y]),
             axis=1)
    return df_merged


def get_consistency(df, feature, distance=d_KL,
                    col='author', col_unique='book_id',
                    suffix_left='', suffix_right='_y', col_name=None):
    df_merged = merge_df(df, feature=feature,
             col_to_merge=col,
             col_unique=col_unique,
             suffix_left=suffix_left, suffix_right=suffix_right)
    df_merged = get_distance_column(df_merged, feature=feature,  
                                    col=col, distance=distance,
                                    suffix_left=suffix_left,
                                    suffix_right=suffix_right,
                                    col_name=col_name)
    return df_merged
    

def return_consistency_data(feature, distance=d_KL,
                            path_consistency='data/pickle/consistency/author',
                            df=None, col='author', col_unique='book_id'):
    df_consistency = try_to_load_as_pickled_object_or_None('{}/df_consitency_{}.p'.format(path_consistency,feature))
    if df_consistency is None:
        if df is None: df = load_corpus()
        df_consistency = get_consistency(df, feature=feature, distance=distance,
                        col=col, col_unique=col_unique)
        save_as_pickled_object(df_consistency, '{}/df_consitency_{}.p'.format(path_consistency,feature))

    return df_consistency


def class_consistency(df_consistency, feature_name, col_name='author'):
    freq_compare = feature_name+'_compare'
    kl_norm_trans_auth = df_consistency.groupby(col_name, as_index=False)\
        .agg({freq_compare: [np.mean, np.std]},axis=1)
    kl_norm_trans_auth.columns = [col_name, freq_compare, freq_compare+'_std']  
    kl_norm_trans_auth.sort_values(freq_compare, ascending=True,inplace=True)
    return kl_norm_trans_auth


def is_normal_consistency(df_consistency, feature_name,
                            col_name='author',
                            nb_simulations=100):
    df_test_normal = pd.DataFrame(df_consistency.groupby(col_name)\
                ['{}_compare'.format(feature_name)].agg([sp.stats.normaltest,
                 sp.stats.shapiro, 'count']))
                    
                    
    df_test_normal.columns=['normaltest', 'shapiro', 'count']
    df_test_normal.reset_index(inplace=True)
  
    df_test_normal['normaltest_p'] = df_test_normal['normaltest'].apply(lambda x:x[1])
    df_test_normal['normaltest_k'] = df_test_normal['normaltest'].apply(lambda x:x[0])  
    
    df_test_normal['shapiro_p'] = df_test_normal['shapiro'].apply(lambda x:x[1])
    df_test_normal['shapiro_k'] = df_test_normal['shapiro'].apply(lambda x:x[0])
    
    return df_test_normal






