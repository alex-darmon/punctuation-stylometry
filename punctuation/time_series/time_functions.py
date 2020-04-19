#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:29:31 2019

@author: alexandradarmon
"""

from punctuation.utils.utils import (
        load_corpus,
        int_or_nan
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from punctuation.config import options
from punctuation.visualisation.visualisation import color_vector, marker_vector


###### Repartition over time deathdate birthdate ###
def bin_date(x):
    if pd.isnull(x):
        return None
    else:
        return int(int(x)/10) * 10
    
def middle_age(b, d):
    if pd.isnull(b) and pd.isnull(d):
        return None
    if not(pd.isnull(b)):
        return b+30
    if not(pd.isnull(d)):
        return d-30
    else:
        return int((int(d)+int(b))/2)


def get_temporal(df=None):
    if df is None:
        df = load_corpus()
        
    df_temporal = df.copy()
    df_temporal['author_birthdate'] = df_temporal['author_birthdate'].apply(int_or_nan)
    df_temporal['author_deathdate'] = df_temporal['author_deathdate'].apply(int_or_nan)
    
    df_temporal['author_birthdate_bin'] = df_temporal['author_birthdate'].apply(bin_date)
    df_temporal['author_deathdate_bin'] = df_temporal['author_deathdate'].apply(bin_date)
    df_temporal['author_middle_age'] = df_temporal[['author_birthdate',
                                                    'author_deathdate']].apply(
            lambda row: middle_age(row[0], row[1]),axis=1)
    df_temporal['author_middle_age_bin'] = df_temporal['author_middle_age'].apply(bin_date)

    return df_temporal



def plot_histogram_years(df_temporal, show_middleyear=True,# col_name='author_birthdate',
                         to_show=True, print_legend=False,
                         show_labels=True):
    
    bar_width=2.5
    col1 = 'author_birthdate_bin'
    auth_date1 = df_temporal\
                 [df_temporal.author_birthdate >=1500]\
                 .dropna(subset=[col1]).\
        groupby(col1, as_index=False)['title'].count()
    auth_date1.sort_values(col1, inplace=True)
    
    col2 = 'author_deathdate_bin'
    auth_date2 = df_temporal\
                [df_temporal.author_birthdate >=1500]\
                .dropna(subset=[col2]).\
        groupby(col2, as_index=False)['title'].count()
    auth_date2.sort_values(col2, inplace=True)
    
    
    col3 = 'author_middle_age_bin'
    auth_date3 = df_temporal\
                 [df_temporal.author_birthdate >=1500]\
                 .dropna(subset=[col3]).\
        groupby(col3, as_index=False)['title'].count()
    auth_date3.sort_values(col3, inplace=True)
    
    
    
    ax = plt.subplot(111, )
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
        
    idx2 = list(map(lambda x: x+2*bar_width, list(auth_date2[col2])))
    idx3 = list(map(lambda x: x+bar_width, list(auth_date3[col3])))
    
    
    plt.bar(list(auth_date1[col1]), list(auth_date1['title']), width = 10,
             color='b', label='birth date'
             )
    
    plt.bar(idx2, list(auth_date2['title']), width = 10,# hatch="/",
            color='grey', label='death date', 
            )
    if show_middleyear:
        plt.bar(idx3, list(auth_date3['title']), width = 10,
                color='y',alpha=0.4, label='middle date'
                )
    if print_legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 1.4, .102),fontsize=18, loc=3,
                   mode="expand", ncol=3, frameon=False)
        plt.xticks( list(np.linspace(0,len(auth_date1)-1, 20)), [str(list(auth_date1[col1])[int(i)]) \
                    for i in list(np.linspace(0,len(auth_date1)-1, 20))], rotation=90)
    if show_labels:    
        plt.xlabel('year')
        plt.ylabel('number of documents')
    #plt.title('Number of books by date in the Database')
    if to_show:
        plt.show()


def plot_col_overtime(df_temporal,col, col_date,
                      min_date=1700, max_date=1950,
                      to_show=True, print_legend=False,
                      confidence=0.95, with_bin=True):
    new_temporal_df = df_temporal[(df_temporal[col_date]>=min_date)&
            (df_temporal[col_date]<=max_date)].dropna(subset=[col_date])
    new_temporal_df[col+'_std'] = new_temporal_df[col]
    new_temporal_df[col+'_sem'] = new_temporal_df[col]
    new_temporal_df[col+'_count'] = new_temporal_df[col]
    new_temporal_df[col+'_min'] = new_temporal_df[col]
    new_temporal_df[col+'_max'] = new_temporal_df[col]
    
    freq_over_time = new_temporal_df.groupby('{}_bin'.format(col_date),as_index=False)\
            .agg({col:np.mean,
              col+'_sem': sp.stats.sem,
              col+'_std': np.std,
              col+'_min': np.min,
              col+'_max': np.max,
              col+'_count': 'count'},axis=1)
    freq_over_time.sort_values('{}_bin'.format(col_date), inplace=True)    
  
    list_mean = list(freq_over_time[col])
    list_std = list(freq_over_time[col+'_std'])
    list_sem = list(freq_over_time[col+'_sem'])
    list_count = list(freq_over_time[col+'_count'])
    list_max = []#list(freq_over_time[col+'_max'])
    list_min = [] #list(freq_over_time[col+'_min'])
    list_lower_bound = []
    list_upper_bound = []

    list_dates = list(freq_over_time['{}_bin'.format(col_date)])
    for mean, cou, se, std in zip(list_mean, list_count, 
                                  list_sem, list_std):
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
    print(len(list_max))
    ax = plt.subplot(111, )
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    axes = plt.gca()
    axes.set_xlim([min_date, max_date])
    ax.plot(list_dates,list_mean, color="grey",
            marker='o', label='mean')
    ax.fill_between(list_dates, list_max, list_min, color="grey", 
                    label='confidence intervals', alpha=0.2)
    
    plt.xlabel('year')
    plt.ylabel('number of words')
    
    plt.xticks(np.arange(min_date, max_date, int((max_date-min_date)/6)+1))
    if print_legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 1, .102),
                   fontsize=15, loc=3,mode="expand", ncol=5, frameon=False)
    if to_show:
        plt.show()




def plot_freq_overtime(df_temporal, list_freq_pun_col, col_date,
                       min_date=1700, max_date=1950,
                       to_show=True, print_legend=False,
                       confidence=0.95, with_bin=True,
                       show_ci=True):
    for i in list_freq_pun_col:
        col = 'FREQ_'+str(i)
        new_temporal_df = df_temporal[(df_temporal[col_date]>=min_date)&
            (df_temporal[col_date]<=max_date)].dropna(subset=[col_date])
        new_temporal_df[col+'_std'] = new_temporal_df[col]
        new_temporal_df[col+'_sem'] = new_temporal_df[col]
        new_temporal_df[col+'_count'] = new_temporal_df[col]
        new_temporal_df[col+'_min'] = new_temporal_df[col]
        new_temporal_df[col+'_max'] = new_temporal_df[col]
        
        freq_over_time = new_temporal_df.groupby('{}_bin'.format(col_date),as_index=False)\
                .agg({col:np.mean,
                  col+'_sem': sp.stats.sem,
                  col+'_std': np.std,
                  col+'_min': np.min,
                  col+'_max': np.max,
                  col+'_count': 'count'},axis=1)
        freq_over_time.sort_values('{}_bin'.format(col_date), inplace=True)    
  
        list_mean = list(freq_over_time[col])
        list_std = list(freq_over_time[col+'_std'])
        list_sem = list(freq_over_time[col+'_sem'])
        list_count = list(freq_over_time[col+'_count'])
        list_max = list(freq_over_time[col+'_max'])
        list_min = list(freq_over_time[col+'_min'])
        list_lower_bound = []
        list_upper_bound = []
    
        list_dates = list(freq_over_time['{}_bin'.format(col_date)])
        for (mean, cou, se, std, 
#             maxi, mini
             ) in zip(list_mean, list_count, 
                                      list_sem, list_std):
#                                      ,list_max, list_min):
            h = se * sp.stats.t.ppf((1 + confidence) / 2., cou-1)
#            list_max.append(mean + h) #(maxi)
#            list_min.append(mean - h) #(mini)
            
    #          if method == 't':
    #        test_stat = stats.t.ppf((interval + 1)/2, n)
    #      elif method == 'z':
            test_stat = sp.stats.norm.ppf((confidence + 1)/2)
            lower_bound = mean - test_stat * std / np.sqrt(cou)
            upper_bound = mean + test_stat * std / np.sqrt(cou)
            list_lower_bound.append(lower_bound)
            list_upper_bound.append(upper_bound)
        
        ax = plt.subplot(111, )
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
    
        axes = plt.gca()
        axes.set_xlim([min_date, max_date])
        ax.plot(list_dates,list_mean, color=color_vector[i],
                label=options.punctuation_vector[int(col[-1])], 
                marker=marker_vector[i])
        if show_ci:
            ax.fill_between(list_dates, list_max, list_min, color="grey", alpha=0.2)
    
    plt.xlabel('year')
    plt.ylabel('frequency')
    
    plt.xticks(np.arange(min_date, max_date, int((max_date-min_date)/6)+1))
    if print_legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 1, .102),
                   fontsize=15, loc=3,mode="expand", ncol=5, frameon=False)
    if to_show:
        plt.show()







#for i in range(0,10):
#    list_freqs = list(temporal_df\
#                      [temporal_df['author_birthdate']>1500] \
#                      .dropna(subset=['author_birthdate'])\
#                          ['FREQ_'+str(i)])
#    plt.plot(list_years, list_freqs, 'o', label='FREQ_'+str(options.punctuation_vector[i]),
#             color = 'b')
#    plt.xlabel('Year')
#    plt.ylabel('Frequency')
#    plt.title('Frequency of punctuation mark '+str(options.punctuation_vector[i])+\
#              ' in books over the birthdate of the author' )
#    plt.legend(loc=0)
#    plt.show()
