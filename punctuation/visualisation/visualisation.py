#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:04:50 2019

@author: alexandradarmon
"""

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from punctuation.config import options
from punctuation.visualisation.heatmap_functions import heatmap, annotate_heatmap
from webcolors import hex_to_rgb

color_vector = ['#2764A2','#EC7428','#438823', '#B9312B', '#785BAD','#72473D',
                '#CA6FB6', '#6C6C6C','#B1AC27', '#44ADC1']

markers = {'o': 'circle','D': 'diamond', 'p': 'pentagon',
            'v': 'triangle_down', '^': 'triangle_up', 
           '<': 'triangle_left', '>': 'triangle_right', 
           's': 'square', '*': 'star','x': 'x',
           
           '_': 'hline', 'p': 'pentagon', 
           'h': 'hexagon1', 'H': 'hexagon2',  'x': 'x',
           'D': 'diamond', 'd': 'thin_diamond', '|': 'vline',  '+': 'plus',
           'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright',
           2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright', 
           6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', '*': 'star', 
           9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 
           'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}

marker_vector =  list(markers.keys())


rgb_color_vector = [hex_to_rgb(i) for i in color_vector]


def get_overall_kdeplot(df,subfile,
                        punctuation_vector=options.punctuation_vector,
                        freq_pun_col=options.freq_pun_col,
                        with_pairs=False):
    
    for col1, pun1 in zip(freq_pun_col, punctuation_vector):
        sns.kdeplot(df[col1], label='{}'.format(pun1), color='black')
        plt.legend(loc=0)
        plt.savefig('results/stats_corpus/{}/kdeplot_{}.png'.format(subfile,col1))
        plt.show()
        if with_pairs:
            for col2, pun2 in zip(freq_pun_col[freq_pun_col.index(col1)+1:],
                            punctuation_vector[punctuation_vector.index(pun1)+1:]):
                sns.kdeplot(df[col1], df[col2], label='{},{}'.format(pun1,pun2))
                plt.legend(loc=0)
                plt.savefig('results/stats_corpus/{}/kdeplot_{}_{}.png'.format(subfile,
                            col1,
                            col2))
                plt.show()


def get_overall_hist(df,subfile,
                        punctuation_vector=options.punctuation_vector,
                        freq_pun_col=options.freq_pun_col):
    bins = np.arange(0,1,0.01)
    for col1, pun1 in zip(freq_pun_col, punctuation_vector):
        ax = plt.subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(options.font_size)
        plt.hist(df[col1], bins=bins, label=pun1, color='blue')
        plt.legend(loc=0, fontsize=options.font_size)
        plt.xlabel('punctuation frequency')
        plt.ylabel('number of documents')
        plt.savefig('results/stats_corpus/{}/hist_{}.png'.format(subfile,col1))
        plt.show()


def show_weapon_hist(kl_within_author_samples, kl_between_author_samples,
                     type_compute_baseline,path_res,feature_name,
                     baseline_between=None,
                     baseline_within=None,
                     bins=100, to_show=True):
    
    bin_size = 0.1
    bins = np.arange(0,2, bin_size)
    x_bins = np.arange(0,2+bin_size, 4*bin_size)
    
    ax = plt.subplot(111)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(options.font_size)
        
        
    y1, bin_edges1=np.histogram(kl_within_author_samples,bins=bins)
    y1 = list(map(lambda x: x/sum(y1), y1))
    bincenters1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
    
    y2, bin_edges2=np.histogram(kl_between_author_samples,bins=bins)
    y2 = list(map(lambda x: x/sum(y2), y2))
    bincenters2 = 0.5*(bin_edges2[1:]+bin_edges2[:-1])
    
#    plt.hist(kl_within_author_samples, bins=bins,  color='black',
#              alpha=0.4,)
    
    plt.bar(bincenters1, y1, width=bin_size,
             color='black',alpha=0.3,)
    plt.plot(bincenters1,y1,'-', color='black')
    
    plt.bar(bincenters1, y2, width=bin_size,
             color='blue',alpha=0.3,)
    plt.plot(bincenters2, y2, '-',  color='blue')
    if type_compute_baseline:
        plt.axvline(baseline_between, color='blue', linestyle=':')
        plt.axvline(baseline_within, color='black', linestyle=':')
    plt.xlim(min(min(kl_within_author_samples), 
                 min(kl_between_author_samples)),2)
    plt.ylim(0,1)
    plt.yticks([0,0.5,1])
    plt.xticks(x_bins)
    
    plt.xlabel('KL divergence')
    plt.ylabel('frequency')
    
    plt.legend('')
    plt.savefig('{}/kl_hist_comparison_{}.png'.format(path_res,feature_name))
    if to_show: plt.show()


## CUMSUM REPRESENTATION
#y1_cum_sum = pd.Series(y1).cumsum()
#y1_cum_sum = y1_cum_sum.tolist()
#
#y2_cum_sum = pd.Series(y2).cumsum()
#y2_cum_sum = y2_cum_sum.tolist()
#
#
#plt.plot(bincenters1, y1_cum_sum, color='black', label='within')
#plt.plot(bincenters1, y2_cum_sum, color='blue', label='between')
#plt.legend()


def plot_list_class(df, class_name='author'):
    res = df.groupby([class_name],as_index=False)\
    ['book_id'].count().rename(columns={'book_id':'nb_books'}).sort_values('nb_books',ascending=False)
#    list_author = list(res[class_name])
#    list_nb_books = list(res['nb_books'])
#
#    plt.bar(list(range(0,len(list_author))), list_nb_books)
#    plt.xticks([10,50,100,150,200],fontsize=options.font_size)
#    plt.yticks([0,20,40,60],fontsize=options.font_size)
#    plt.xlim([10,230])
#    plt.xlabel('Number of documents',fontsize=options.font_size)
#    plt.ylabel('Number of {}s'.format(class_name),fontsize=options.font_size)
#    plt.bar(list_nb_books, list_nb_authors,width=3, color='blue')
#    
    
    
    res = res[[class_name,'nb_books']].\
            groupby('nb_books',as_index=False)[class_name].count()
    
    
    list_nb_authors = list(res[class_name])
    list_nb_books = list(res['nb_books'])
    
    
    ax = plt.subplot(111, )
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(options.font_size)
    ax.set_xlim([10,230])
    
    plt.xticks([10,50,100,150,200],fontsize=options.font_size)
    plt.yticks([0,20,40,60],fontsize=options.font_size)
    plt.xlim([10,230])
    plt.xlabel('number of documents',fontsize=options.font_size)
    plt.ylabel('number of {}s'.format(class_name),fontsize=options.font_size)
    plt.bar(list_nb_books, list_nb_authors,width=3, color='blue')
    plt.show()


def plot_hist_punc(freq, punctuation_vector=options.punctuation_vector):
    y = freq
    x = list(range(0,10))
    
    ax = plt.subplot(111)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(options.font_size)
    ax.bar(x, y, align='center', color='b') #< added align keyword
    ax.xaxis_date()
    ax.set_ylim(bottom=0, top=0.7)
    plt.xticks(list(range(0,10)), punctuation_vector[:-1]+['...'])
    
    plt.show()


def plot_hist_words(freq):
    ax = plt.subplot(111, )
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(options.font_size)
    plt.rcParams.update({'font.size': options.font_size})
    plt.bar(list(range(0,len(freq))), freq, color='magenta', align='center')
    ax.set_ylim(bottom=0, top=0.4)
    #plt.xticks(list(range(0,len(freq))), punctuation_vector)
    plt.show()

def func(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")


def plot_trans_mat(mat_nb_words,
                   punctuation_vector=options.punctuation_vector):
    vegetables = punctuation_vector[:-1]+['...']
    farmers = punctuation_vector[:-1]+['...']
    
    harvest = np.array(mat_nb_words)
    
    
    fig, ax = plt.subplots()
    im, _ = heatmap(harvest, vegetables, farmers, ax=ax,
                    )
    
    annotate_heatmap(im, valfmt="{x:.1f}", size=7)
    plt.tight_layout()
    plt.show()


def plot_scatter_freqs(df, title1=None, title2=None,
                       freq1=None, freq2=None,
                       font_size=options.font_size,
                       ):
    if title1 is None:
        title1 = random.choice(df['title'].tolist())
    if title2 is None:
        title2 = random.choice(df['title'].tolist())
    
    if freq1 is None:
        freq1 = df[df['title']==title1]['freq_pun'].iloc[0]
    
    if freq2 is None:
        freq2 = df[df['title']==title2]['freq_pun'].iloc[0]
    
    
    ax = plt.subplot(111, )
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(options.font_size)
        
    plt.xlabel('$\it{'+title1.replace(' ','\ ')+'}$', fontsize=options.font_size)
    plt.ylabel('$\it{'+title2.replace(' ','\ ')+'}$', fontsize=options.font_size)
    plt.gca().set_aspect('equal', adjustable='box')
    
    vect = np.linspace(-0, 0.5, 10)
    plt.xticks([-0.,0.25,0.5], ['0', '0.25', '0.5'], fontsize=options.font_size)
    plt.yticks([-0,0.25,0.5],['0', '0.25', '0.5'], fontsize=options.font_size)
    for i in range(0,len(color_vector)):   
        plt.plot(freq1[i], freq2[i],  color=color_vector[i], marker="o")
    plt.plot(vect, vect, color = 'black', alpha=0.2)
    plt.show()
