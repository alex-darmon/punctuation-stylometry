#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:35:55 2019

@author: alexandradarmon
"""

import pickle
import numpy as np
import pandas as pd
import sys
import os
import logging
from string import ascii_letters
from punctuation.config import options
from multiprocessing import Pool, cpu_count
from logs.logger import logging_function # Inherit the logger from the calling function
logger = logging.getLogger(__name__)


@logging_function(logger)
def trial():
    print('Success!')


@logging_function(logger)
def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

@logging_function(logger)
def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj

def load_corpus(path='data/pickle/corpus_features.p'):
    res = try_to_load_as_pickled_object_or_None(path)
    if res is not None:
        return res
    if 'genre' in path:
        res = define_genre(path=path)
        return res
    if 'author' in path:
        #TODO
        pass

def define_genre(df=None, genres_to_include=options.genres_to_include,
                 path='data/pickle/genre_features.p'):
    if df is None: df=load_corpus()
    genre_df = df[df.genre.isin(genres_to_include)]
    save_as_pickled_object(genre_df, path)
    return genre_df


def chunks(l, n):
   n = max(1, n)
   return list(l[i:i+n] for i in range(0, len(l), n))


@logging_function(logger)
def splitter_function(txt, first_term, last_term):
    if txt is None:
        return None
    if first_term in txt and last_term in txt:
        return txt.split(first_term)[1].split(last_term)[0]
    else:
        return None
    
def wrap_function(list_elts, fun):
    list_res = []
    for x in list_elts:
        try:
            res = fun(x)
        except:
            res = None
        list_res.append(res)
    return list_res

def wrap_pool(list_elts, fun):
    total_threads = cpu_count()
    chunk_size = int(len(list_elts) / total_threads) + 1
    sets_to_be_computed = chunks(list_elts, chunk_size)
    pool = Pool(total_threads)
    results = pool.map(lambda x: wrap_function(x, fun=fun), sets_to_be_computed)
    l_final_res = []
    for l_res in results:
        l_final_res = l_final_res+l_res
    pool.close()
    pool.join()
    return l_final_res


def int_or_nan(x):
    try:
        x = int(x)
    except:
        x = np.nan
    return x
