#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:45:01 2019

@author: alexandradarmon
"""

import sys
import pandas as pd
import numpy as np
from itertools import product
from punctuation.config import options
from punctuation.visualisation.visualisation import (
plot_hist_punc, plot_trans_mat, plot_hist_words,plot_list_class,
get_overall_kdeplot, plot_scatter_freqs, get_overall_hist,
)
from punctuation.utils.utils import load_corpus


def plot_features(df):
    plot_hist_punc(df[options.freq_pun_col].mean())
    plot_hist_words(df[options.freq_nb_words_col].mean())
    plot_hist_words(df[options.freq_length_sen_with_col].mean())
    
    plot_trans_mat(np.reshape(df[options.mat_nb_words_pun_col].mean().as_matrix(), (10,10)),
                       punctuation_vector=options.punctuation_vector)
    plot_trans_mat(np.reshape(df[options.transition_mat_col].mean().as_matrix(), (10,10)),
                       punctuation_vector=options.punctuation_vector)
    plot_trans_mat(np.reshape(df[options.norm_transition_mat_col].mean().as_matrix(), (10,10)),
                       punctuation_vector=options.punctuation_vector)
    
    
    plot_scatter_freqs(df, title1=None, title2=None,
                       freq1=None, freq2=None, font_size=18)


full_run = False


### AUTHOR ANALYSIS
df = load_corpus()

sys.exit(2)

plot_list_class(df, class_name='author')
plot_list_class(df, class_name='genre')
plot_list_class(df, class_name='author_birthdate')


#4. corpus overall info
plot_features(df)


get_overall_hist(df, subfile='freq_pun',
                    punctuation_vector=options.punctuation_vector[:-1]+['...'],
                    freq_pun_col=options.freq_pun_col)

if full_run:
    get_overall_kdeplot(df,subfile='freq_pun',
                        punctuation_vector=options.punctuation_vector[:-1]+['...'],
                        freq_pun_col=options.freq_pun_col,
                        with_pairs=False)
    
    get_overall_kdeplot(df,subfile='trans_mat',
                        punctuation_vector=list(product(options.punctuation_vector, repeat=2)),
                        freq_pun_col=options.transition_mat_col,
                        with_pairs=False)
    
    get_overall_kdeplot(df,subfile='mat_nb_words',
                        punctuation_vector=list(product(options.punctuation_vector, repeat=2)),
                        freq_pun_col=options.mat_nb_words_pun_col,
                        with_pairs=False)


### GENRE ANALYSIS
df_genre = load_corpus(path='data/pickle/corpus_features.p')

plot_list_class(df_genre, class_name='author')
plot_list_class(df_genre, class_name='genre')
plot_list_class(df_genre, class_name='author_birthdate')

plot_features(df_genre)

if full_run:
    get_overall_kdeplot(df_genre,subfile='freq_pun',
                        punctuation_vector=options.punctuation_vector,
                        freq_pun_col=options.freq_pun_col,
                        with_pairs=False)
    
    get_overall_kdeplot(df_genre,subfile='trans_mat',
                        punctuation_vector=list(product(options.punctuation_vector, repeat=2)),
                        freq_pun_col=options.transition_mat_col,
                        with_pairs=False)
    
    get_overall_kdeplot(df_genre,subfile='mat_nb_words',
                        punctuation_vector=list(product(options.punctuation_vector, repeat=2)),
                        freq_pun_col=options.mat_nb_words_pun_col,
                        with_pairs=False)

# TEMPORAL ANALYSIS
#
#print(len(temporal_df
#          #[temporal_df.author_birthdate <1500]\
#          .dropna(subset=['author_birthdate']).drop_duplicates('author')))
#
#print(len(temporal_df[temporal_df.author_birthdate >=1500]\
#          .dropna(subset=['author_birthdate']).drop_duplicates('author')))
#print(len(temporal_df[temporal_df.author_deathdate >=1500]\
#          .dropna(subset=['author_deathdate']).drop_duplicates('author')))
#
#print(len(temporal_df[temporal_df.author_middle_age >=1500]\
#          .dropna(subset=['author_middle_age']).drop_duplicates('author')))

