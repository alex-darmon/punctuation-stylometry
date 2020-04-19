#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:51:59 2018

@author: alexandra.darmon
"""

import scipy as sp
import numpy as np
from punctuation.feature_operations.distances import d_KL
from punctuation.utils.utils import (
        try_to_load_as_pickled_object_or_None, 
        load_corpus,
        save_as_pickled_object)
import scipy.io


def convert_to_matlab(distance_matrix, list_auths, name):
        sp.io.savemat(name+'.mat', mdict={'distance_matrix': distance_matrix})
        sp.io.savemat(name+"_list_authors.mat", mdict={'list_authors': list_auths})

def get_distance_matrices(list_auth, feature,feature_name, df=None, 
                          distance=d_KL,
                          col_name='author',
                          path_distance_matrices='data/pickle/distance_matrices/author'):
    distance_matrix = try_to_load_as_pickled_object_or_None(
            '{}/dmat_{}_{}.p'.format(path_distance_matrices,
                           len(list_auth),feature_name))
    list_auths = try_to_load_as_pickled_object_or_None(
            '{}/list_auth_{}_{}.p'.format(path_distance_matrices,
                           len(list_auth),feature_name))
    
    if distance_matrix is None or list_auths is None:
        
        if df is None: 
            df = load_corpus()
        df_auth = df[df[col_name].isin(list_auth)]
        df_auth[col_name] = df_auth[col_name].astype("category")
        df_auth[col_name].cat.set_categories(list_auth, inplace=True)
        df_auth.sort_values([col_name], inplace=True)
        
        distance_matrix = np.zeros((len(df_auth),len(df_auth)), dtype='f')
        for i in range(0,len(df_auth)):
            for j in range(0, len(df_auth)):
                res1 =  df_auth[feature].iloc[i]
                res2 =  df_auth[feature].iloc[j]
                res = distance(res1, res2)
                distance_matrix[i,j] = res
        list_auth = list(df_auth[col_name].drop_duplicates())
        list_auths = [list_auth.index(i) for i in list(df_auth[col_name])]
        save_as_pickled_object(distance_matrix, '{}/dmat_{}_{}.p'.format(path_distance_matrices,
                               len(list_auth),feature_name))
        save_as_pickled_object(list_auths, '{}/list_auth_{}_{}.p'.format(path_distance_matrices,
                               len(list_auth),feature_name))
    
    convert_to_matlab(distance_matrix, list_auths,
                      '{}/matlab_{}_{}'.format(path_distance_matrices,
                           len(list_auth),feature_name))
    return distance_matrix, list_auths