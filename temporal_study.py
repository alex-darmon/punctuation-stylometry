#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:25:08 2019

@author: alexandradarmon
"""

### RUN TIME SERIES


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
        int_or_nan
)
from punctuation.time_series.time_functions import (
    get_temporal,
    plot_histogram_years,
    plot_freq_overtime,
    plot_col_overtime
)
import pandas as pd
import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn-paper')


df = load_corpus()
df_temporal = get_temporal(df=df)


plot_histogram_years(df_temporal, show_middleyear=False,
                     to_show=True, print_legend=False)

plot_histogram_years(df_temporal,show_middleyear=True,
                     to_show=True, print_legend=False)

list_freq_pun_col = list(range(options.nb_signs))

freq_pun_col_1 = [1,4,5]
freq_pun_col_2 = [0,7]
freq_pun_col_3 = [2,3,6,8,9]


for f in [freq_pun_col_1,freq_pun_col_2,freq_pun_col_3]:
    plot_freq_overtime(df_temporal, f,
                   col_date='author_middle_age',
                   min_date=1700, max_date=1950,
                   to_show=True, print_legend=True)

plot_freq_overtime(df_temporal, list_freq_pun_col,
                   col_date='author_middle_age',
                   min_date=1700, max_date=1950,
                   to_show=True, print_legend=False)


wells = pd.read_csv('data/Marya_Wells.csv').sort_values('Date')
wells = pd.merge(wells, df_temporal, how='inner', on='title')

wells['Date_bin'] = wells['Date']
plot_freq_overtime(wells, list_freq_pun_col,
                   col_date='Date',
                   min_date=min(wells['Date']),
                   max_date=1922,
                   print_legend=False, show_ci=True)


fleming = pd.read_csv('data/Alex_Fleming.csv').sort_values('Date')
fleming = pd.merge(fleming, df_temporal, how='left', on='title')

fleming['Date_bin'] = fleming['Date']
plot_freq_overtime(fleming, list_freq_pun_col,
                   col_date='Date',
                   min_date=min(fleming['Date']),
                   max_date=max(fleming['Date']),
                   print_legend=False, show_ci=True)



shakespeare = pd.read_csv('data/Alex_Shakespeare.csv').sort_values('Date')
shakespeare = pd.merge(shakespeare, df_temporal, how='inner', on='title')

shakespeare['Date_bin'] = shakespeare['Date']
plot_freq_overtime(shakespeare, list_freq_pun_col,
                   col_date='Date',
                   min_date=min(shakespeare['Date']),
                   max_date=max(shakespeare['Date']),
                   print_legend=False)



dickens = pd.read_csv('data/Alex_Dickens.csv').sort_values('Date')
dickens = pd.merge(dickens, df_temporal, how='left', on='title')

dickens['Date_bin'] = dickens['Date']
plot_freq_overtime(dickens, list_freq_pun_col,
                   col_date='Date',
                   min_date=1836,
                   max_date=1871,
                   print_legend=False, show_ci=True)






#### FIGURE 


def get_average_nb_words_by_pun(l):
    sum_words = 0
    sum_punc = 0
    for c in l:
        if c in options.punctuation_vector:
            sum_punc+=1
        elif type(c) == int:
            sum_words+=c
        else:
            print('Stop')
            print(c)
    return (sum_words, sum_punc)
            
new_temporal_df = df_temporal.dropna(subset=['author_middle_age'])
new_temporal_df['nb_words_nb_pun'] = new_temporal_df['seq_nb_words'].apply(get_average_nb_words_by_pun)
new_temporal_df['nb_words/nb_pun'] = new_temporal_df['nb_words_nb_pun'].apply(lambda l:l[0]/l[1])

col = 'nb_words/nb_pun'


plot_col_overtime(new_temporal_df, col, col_date='author_middle_age',
                      min_date=1700, max_date=1950,
                      to_show=True, print_legend=False,
                      confidence=0.95, with_bin=True)
