#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:01:30 2019

@author: alexandradarmon
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from punctuation.feature_operations.distances import d_KL, pearson
from punctuation.consistency.consistency_computation import (
        return_consistency_data,
)
from punctuation.consistency.consistency_visualisation import (
        compute_baseline
)
import scipy as sp
from punctuation.config import options
from punctuation.utils.utils import  load_corpus
from punctuation.visualisation.visualisation import (show_weapon_hist)
color1 = 'gist_earth'
color2 = 'gist_stern'


##############################
#THE WEAPON

def show_weapon(feature_name,
                col_name='author', nb_books=1000,
                df=None, distance=d_KL, col_unique='book_id',
                path_consistency='data/pickle/consistency/author',
                path_res='results/comparison/author',
                reshuffle=False,
                type_compute_baseline=None,
                to_show=True):
    
    baseline_between = None 
    baseline_within = None
    if df is None:  df=load_corpus()
    
    df_consistency_within = return_consistency_data(feature_name,
                                                    path_consistency=path_consistency,
                                                    distance=distance,
                                                    df=df,
                                                    col=col_name, 
                                                    col_unique=col_unique)


    baseline_between, df_consistency_between = \
            compute_baseline(feature_name, df=df, distance=distance, 
                         nb_books=nb_books, col_unique=col_unique,
                         col_class=col_name,
                         path_consistency=path_consistency,
                         reshuffle=reshuffle)
    
    if type_compute_baseline:
        baseline_between = type_compute_baseline(df_consistency_between['{}_compare'.format(feature_name)].tolist())
        baseline_within = type_compute_baseline(df_consistency_within['{}_compare'.format(feature_name)].tolist())
    
    kl_within_author_samples = df_consistency_within['{}_compare'.format(feature_name)].tolist()
    kl_between_author_samples = df_consistency_between['{}_compare'.format(feature_name)].tolist()
    show_weapon_hist(kl_within_author_samples, kl_between_author_samples,
                     type_compute_baseline,path_res,feature_name,
                     baseline_between=baseline_between,
                     baseline_within=baseline_within,
                     bins=100, to_show=to_show)
    print(feature_name)
    print(sp.stats.ks_2samp(kl_within_author_samples, kl_between_author_samples))
    return (df_consistency_between, df_consistency_within, 
            baseline_between, baseline_within)


##############################

# AUTHOR 

df=load_corpus()
for feature_name in options.feature_names:
    (df_consistency_between, df_consistency_within, 
     baseline_between, baseline_within) = \
         show_weapon(feature_name,
                col_name='author', nb_books=1000,
                df=df, distance=d_KL, col_unique='book_id',
                path_consistency='data/pickle/consistency/author',
                path_res='results/comparison/author',
                reshuffle=False,
                type_compute_baseline=np.mean,
                to_show=True)
    print('{}: baseline_between {}, baseline_within {}'.format(
           feature_name,  baseline_between, baseline_within))
    


##############################

# GENRE  

df_genre = load_corpus('data/pickle/genre_features.p')
for feature_name in options.feature_names:
    (df_consistency_between, df_consistency_within, 
     baseline_between, baseline_within) = \
         show_weapon(feature_name,
                col_name='genre', nb_books=1000,
                df=df, distance=d_KL, col_unique='book_id',
                path_consistency='data/pickle/consistency/genre',
                path_res='results/comparison/genre',
                reshuffle=False,
                type_compute_baseline=np.mean,
                to_show=True)
    print('{}: baseline_between {}, baseline_within {}'.format(
           feature_name,  baseline_between, baseline_within))
    
#from scipy.stats import ttest_ind
#tset, pval = ttest_ind(kl_between_author_samples,kl_within_author_samples)
#print("p-values",pval)
#if pval < 0.05:    # alpha value is 0.05 or 5%
#   print(" we are rejecting null hypothesis")
#else:
#  print("we are accepting null hypothesis")
#
#from statsmodels.stats import weightstats as stests
#ztest ,pval1 = stests.ztest(kl_between_author_samples, 
#                            x2=kl_within_author_samples,value=0,alternative='two-sided')
#print(float(pval1))
#if pval<0.05:
#    print("reject null hypothesis")
#else:
#    print("accept null hypothesis")
#


