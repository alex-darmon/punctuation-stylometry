#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:03:19 2019

@author: alexandradarmon
"""


from sklearn.model_selection import train_test_split
from random import sample
from punctuation.utils.utils import (
        try_to_load_as_pickled_object_or_None, 
        load_corpus,
        save_as_pickled_object)
import pandas as pd


def get_train_test_index(nb_classes, df,
                        col_class='author',
                        test_size=0.2, random_state=8,
                        mode_selection=1):
    ''' mode_selection:
    1: uniform
    2: list classes weighted by number of documents in the classes
    '''
    if mode_selection==1:
        list_author = df[col_class].drop_duplicates().tolist()
    if mode_selection==2:
        list_author = df[col_class].tolist()
    first_list = sample(list_author, nb_classes)
    df_class = df[df[col_class].isin(first_list)]
    X = df_class
    y = df_class[col_class]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    return (nb_classes, list(X_train.index), list(X_test.index))
#    return X_train, X_test, y_train, y_test


def get_nn_indexes(size_experiments, df=None,
                col_class='author',
                test_size=0.2, random_state=8,
                save=True,
                path_index_nn = 'data/pickle/recognition/author',
                mode_selection=1):
    df_index_nn = try_to_load_as_pickled_object_or_None('{}/df_index_{}.p'.format(path_index_nn,col_class))
    
    if df_index_nn is None:
        if df is None: df=load_corpus()
        idxex = []
        for nb_classes in size_experiments:
            (nb_classes,
             index_train,
             index_test) = get_train_test_index(nb_classes, df,
                                                col_class=col_class,
                                                test_size=test_size,
                                                random_state=random_state,
                                                mode_selection=mode_selection)
            idxex.append((nb_classes, index_train, index_test))
        df_index_nn = pd.DataFrame(idxex, columns=['nb_classes', 'index_train',
                                                   'index_test'])
        if save:    
            save_as_pickled_object(df_index_nn,
                               '{}/df_index_{}.p'.format(path_index_nn,col_class))
    return df_index_nn


