#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 03:44:16 2019

@author: alexandradarmon
"""

import time
import os
from logs.logger import create_logger
from punctuation.utils.utils import trial
from punctuation.parser.gutenberg_cache_parser import get_cache_info
from punctuation.config import options
import pandas as pd
from multiprocessing import Pool, cpu_count
from punctuation.parser.gutenberg_parser import (
        get_gutenberg_text_tokens_pool,
        get_gutenberg_texts_tokens
    )

import sys
sys.exit(2)

BASEDIR = os.path.join(os.path.dirname('__file__'))
logger = create_logger()

df_cache = pd.read_pickle('data/pickle/cache_step1.p')
total_list_book_ids = list(df_cache['book_id'])

list_book_ids = list(set(total_list_book_ids).difference(
        set(map(lambda x: x.split('.txt')[0],
                         os.listdir('/Users/alexandradarmon/gutenberg_data/text/')))\
                            ))
                            

for l in range(1150, len(total_list_book_ids), 10):
    #print('here1')
    list_book_ids = total_list_book_ids[l:l+10]
    #print('here2')
    if l%50==0: print(l)
                             
    l_text, l_tokens =  get_gutenberg_texts_tokens(list_book_ids)
#    get_gutenberg_text_tokens_pool(list_book_ids, save_text=True)
    print('here4')
    df_text = pd.DataFrame(zip(list_book_ids, l_text, l_tokens), 
                           columns=['book_id','text', 'tokens_nb_words'])
    df_text = pd.merge(df_cache, df_text, on='book_id')
    df_text.to_pickle('data/pickle/gutenberg_text_token/text_token_{}.p'.format(str(l)))
#    print('here5')

#df_res['seq_pun'] =  df_res['seq_nb_words'].apply(seq_pun_only)
#wrap_pool(list_elts, fun)