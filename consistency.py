#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:33:43 2019

@author: alexandradarmon
"""

# 5.1. heatmaps (KL or other distance?)
#import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
#import numpy as np
from punctuation.feature_operations.distances import d_KL, pearson
from punctuation.feature_operations.distance_matrices import get_distance_matrices
from punctuation.consistency.consistency_computation import (
        return_consistency_data,
        class_consistency
)
from punctuation.consistency.consistency_visualisation import (
        get_all_baselines,
        plot_consitency,
        show_random_consistency
)

from punctuation.config import options
from punctuation.utils.utils import  load_corpus


color1 = 'gist_earth'
color2 = 'gist_stern'




def run_consistency(df=None, col_name='author', distance=d_KL,
                    path_consistency='data/pickle/consistency/author',
#                    path_consistency='results/consistency/author',
                    path_distance_matrices='data/pickle/distance_matrices/author',
                    col_unique='book_id', show_distance_matrices=True,
                    features=options.feature_names):

    if df is None:
        if col_name=='author': df = load_corpus()
        if col_name=='genre': df = load_corpus('data/pickle/genre_features.p')
    
    list_consistency_data = []
    for feature_name in features:
        df_consistency_data = return_consistency_data(feature_name, 
                                                      path_consistency=path_consistency,
                                                      distance=distance,
                                                      df=df,
                                                      col=col_name, 
                                                      col_unique=col_unique)
        list_consistency_data.append(df_consistency_data)
#        show_random_consistency(df_consistency_data, feature_name,
#                            col_name=col_name,
#                            nb_simulations=100)
        

    distance_matrices = []
    list_auths = []
    args = []
    
    ### Get the baseline for each feature if exists or compute is otherwise
    if show_distance_matrices:
        for i in range(0,len(features)):
            df_consistency = list_consistency_data[i]
            feature_name  = options.feature_names[i]
            feature = options.features[i]
            kl_norm_trans_auth = class_consistency(df_consistency, feature_name,
                                                    col_name=col_name)
        
            for size in [10, 50]:
                list_auth  = list(kl_norm_trans_auth[col_name].iloc[0:size])
                distance_matrix, list_auths = get_distance_matrices(
                        list_auth, 
                        feature,feature_name, 
                        df,col_name=col_name,
                        path_distance_matrices=path_distance_matrices)
                
                distance_matrices.append(distance_matrix)
                list_auths.append((list_auth, list_auths))
                args.append((feature_name, size))
                
                
                sns.heatmap(distance_matrix, cmap=color1)
                plt.show()
        
    
    ### Get the baseline for each feature if exists or compute it otherwise
    df_baseline = get_all_baselines(features, df=df,
                     nb_books=1000, col_unique=col_unique,
                     col_class=col_name, distance=distance,
                     path_consistency=path_consistency)

    ### Plot consistency for each features with baseline
    for i in range(0, len(features)):
        df_consistency = list_consistency_data[i]
        feature_name = features[i]
        print(feature_name)
        baseline = df_baseline[df_baseline['feature']==feature_name]['baseline'].iloc[0]
        plot_consitency(feature_name, df_consistency,  baseline,
                        with_legend=False, with_title=True,
                        col_name=col_name,
                        path_consistency=path_consistency)


### AUTHOR CONSISTENCY
df = load_corpus()
run_consistency(df=df, col_name='author', distance=d_KL,
                path_consistency='data/pickle/consistency/author',
                path_distance_matrices='data/pickle/distance_matrices/author',
                col_unique='book_id', show_distance_matrices=False,
                features=options.feature_names)


### GENRE CONSISTENCY 
df_genre = load_corpus('data/pickle/genre_features.p')
run_consistency(df=df_genre, col_name='genre', distance=d_KL,
                path_consistency='data/pickle/consistency/genre',
                path_distance_matrices='data/pickle/distance_matrices/genre',
                col_unique='book_id', show_distance_matrices=False,
                features=options.feature_names)


### AUTHOR CONSISTENCY -- PEARSON CORRELATION
df = load_corpus()
run_consistency(df=df, col_name='author', distance=pearson,
                path_consistency='data/pickle/consistency/author_pearson',
                path_distance_matrices='data/pickle/distance_matrices/author_pearson',
                col_unique='book_id', show_distance_matrices=False,
                features=options.feature_names[-1:])

    
    
    
    