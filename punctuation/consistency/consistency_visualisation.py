#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:46:05 2019

@author: alexandradarmon
"""

from punctuation.config import options
import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
from punctuation.feature_operations.distances import d_KL
from punctuation.utils.utils import (
        try_to_load_as_pickled_object_or_None, 
        load_corpus,
        save_as_pickled_object)
import random
import secrets
import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn-white')



def plot_consitency(feature, df_consistency,  baseline, col_name='author',
                    with_legend=False, with_title=False,
                    path_consistency='results/consistency/author',
                    to_show=True,
                    confidence=0.95):
    
    col = feature+'_compare'
    df_consistency[col+'_count'] = df_consistency[col]
    df_consistency[col+'_std'] = df_consistency[col]
    df_consistency[col+'_sem'] = df_consistency[col]
    
    df_consistency_group = df_consistency.groupby(col_name)\
        .agg({col:np.mean,
              col+'_sem': sp.stats.sem,
              col+'_std': np.std,
              col+'_count': 'count'},axis=1)
        
    df_consistency_group.sort_values(col, ascending=False,inplace=True)
    
    list_mean = list(df_consistency_group[col])
    list_std = list(df_consistency_group[col+'_std'])
    list_sem = list(df_consistency_group[col+'_sem'])
    list_count = list(df_consistency_group[col+'_count'])
    list_max = []
    list_min = []
    list_lower_bound = []
    list_upper_bound = []
    list_auth = range(0,len(df_consistency_group))
    
    for mean, cou, se, std in zip(list_mean, list_count, list_sem, list_std):
        h = se * sp.stats.t.ppf((1 + confidence) / 2., cou-1)
        list_max.append(mean + h)
        list_min.append(mean - h)
        
#          if method == 't':
#        test_stat = stats.t.ppf((interval + 1)/2, n)
#      elif method == 'z':
        test_stat = sp.stats.norm.ppf((confidence + 1)/2)
        lower_bound = mean - test_stat * std / np.sqrt(cou)
        upper_bound = mean + test_stat * std / np.sqrt(cou)
        list_lower_bound.append(lower_bound)
        list_upper_bound.append(upper_bound)
        
    
    mpl.rcParams.update({'font.size': options.font_size})
    fig, ax = plt.subplots()
    axes = plt.gca()
    axes.set_xlim([0,len(df_consistency_group)])
    axes.set_ylim([0,1])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(options.font_size)

    ax.plot(list_auth,list_mean,  color='black', label="mean ",)
    
    ax.fill_between(list_auth, list_max, list_min, color="grey", 
                    alpha=0.5, label='confidence intervals')
    ax.plot(list_auth, [baseline]*len(list_auth), '--',
            color='blue', label='baseline')
    if with_legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 2, .102),fontsize=options.font_size, 
                   loc=3,mode="expand", ncol=3, frameon=False)
    plt.yticks([0,0.5,1])
    if path_consistency is not None:
        plt.savefig(path_consistency+'/plot_consistency_{}.png'.format(feature))
    
    if with_title:
        plt.xlabel(col_name)
        plt.ylabel('consistency')
    if to_show: plt.show()

#    ax.plot(list_auth, [0]*len(list_auth),  color='black')
#    plt.axhline(0, color='black')



#def get_baseline_kl(feature, distance, nb_books=1000, text_and_books_df=text_and_books_df):
#    list_books = random.sample(list(text_and_books_df['book_id']), nb_books)
#    selection_df = text_and_books_df[text_and_books_df.book_id.isin(list_books)]
#    selection_df['to_merge'] = 1
#    new_selection_df = pd.merge(selection_df[['to_merge',feature,'book_id','author']],
#                     selection_df[['to_merge',feature,'book_id', 'author']],
#                     on='to_merge')
#    new_selection_df = new_selection_df[new_selection_df['author_x']!=new_selection_df['author_y']]
#
#    feature_x = feature+'_x'
#    feature_y = feature+'_y'
#    new_selection_df[feature+'_compare'] = new_selection_df.apply(lambda row: \
#             distance(row[feature_x], row[feature_y]),
#             axis=1)
#    return new_selection_df[feature+'_compare'].mean()


def get_merged_data(feature, df=None,
                    col_unique='book_id',
                    col_class='author', 
                    path_consistency='data/pickle/consistency'):
     
    
    new_selection_df = \
    try_to_load_as_pickled_object_or_None('{}/df_consistency_baseline_{}.p'.\
                                          format(path_consistency,
                                                 feature))
    if new_selection_df is None:
        if df is None: df = load_corpus()
        selection_df = df.copy()
        selection_df['to_merge'] = 1
        new_selection_df = pd.merge(selection_df[['to_merge',feature,
                                                  col_unique, col_class]],
                         selection_df[['to_merge',feature, col_unique, col_class]],
                         on='to_merge')
        
        save_as_pickled_object(new_selection_df, '{}/df_consistency_baseline_{}.p'.\
                                          format(path_consistency,
                                                 feature))
    return new_selection_df


def compute_baseline(feature, df=None, distance=d_KL, 
                     nb_books=1000, col_unique='book_id',
                     col_class='author',
                     path_consistency='data/pickle/consistency/author',
                     reshuffle=False):
    
    if df is None: df = load_corpus()
    
    new_selection_df = \
    try_to_load_as_pickled_object_or_None('{}/df_consistency_baseline_{}_compared.p'.\
                                          format(path_consistency,
                                                 feature))
    
    if new_selection_df is None or reshuffle:
        new_selection_df = get_merged_data(feature, df=df,
                        col_unique=col_unique,
                        col_class=col_class, 
                        path_consistency=path_consistency)
        
        new_selection_df = \
            new_selection_df[new_selection_df['{}_x'.format(col_class)]\
                                !=new_selection_df['{}_y'.format(col_class)]]
    
        list_books = random.sample(list(new_selection_df.index), nb_books)
        new_selection_df = new_selection_df[new_selection_df.index.isin(list_books)]
        feature_x = feature+'_x'
        feature_y = feature+'_y'
        new_selection_df[feature+'_compare'] = new_selection_df.apply(lambda row: \
                 distance(row[feature_x], row[feature_y]),
                 axis=1)
        reshuffle_index = secrets.token_hex(4) if reshuffle else ''
        save_as_pickled_object(new_selection_df, '{}/df_consistency_baseline_{}_compared{}.p'.\
                                              format(path_consistency,
                                                     feature,
                                                     reshuffle_index))
    
    baseline = new_selection_df[feature+'_compare'].mean()
    print(str((feature, nb_books, baseline, col_class)),
              file=open('{}/baseline_{}.txt'.format(path_consistency,
                        col_class),'a'))
   
    return baseline, new_selection_df


def get_baseline(feature, df=None, distance=d_KL, 
                     nb_books=1000, col_unique='book_id',
                     col_class='author',
                     path_consistency='data/pickle/consistency/author'):
    
    df_baseline = \
    try_to_load_as_pickled_object_or_None('{}/all_consistency_baseline.p'.\
                                          format(path_consistency))
    baseline = None
    if df_baseline is not None:
        try:
            baseline = df_baseline[(df_baseline['nb_books']==nb_books)&
                (df_baseline['col_class']==col_class)&
                (df_baseline['feature']==feature)]['baseline'].iloc[0]
        except:
            baseline = None
    if baseline is None:
        baseline, _ = compute_baseline(feature, df=df, distance=distance, 
                     nb_books=nb_books, col_unique=col_unique,
                     col_class=col_class,
                     path_consistency=path_consistency)
        
    return baseline


def get_all_baselines(features, df=None, distance=d_KL, 
                     nb_books=1000, col_unique='book_id',
                     col_class='author',
                     path_consistency='data/pickle/consistency/author'):
    
    list_baselines = []
    for feature in features:
        baseline  = get_baseline(feature, df=df, distance=distance, 
                     nb_books=nb_books, col_unique=col_unique,
                     col_class=col_class,
                     path_consistency=path_consistency)
        list_baselines.append(baseline)

    df_baseline = pd.DataFrame(list_baselines, columns=['baseline'])
    df_baseline['col_class'] = col_class
    df_baseline['nb_books'] = nb_books
    df_baseline['feature'] = features
    
    save_as_pickled_object(df_baseline, '{}/all_consistency_baseline.p'.\
                                      format(path_consistency))
    return df_baseline



def show_random_consistency(df_consistency, feature_name,
                            col_name='author',
                            nb_simulations=100):
    list_classes = df_consistency[col_name].drop_duplicates().tolist()
    for l in range(nb_simulations):
        class_id = random.choice(list_classes)
        list_kl = df_consistency[df_consistency[col_name]==class_id]\
                ['{}_compare'.format(feature_name)].tolist()
        sns.kdeplot(list_kl)
    plt.show()
