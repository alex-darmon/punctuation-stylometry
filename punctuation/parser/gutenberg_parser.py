#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:46:46 2019

@author: alexandradarmon
"""

import logging
import numpy as np
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from punctuation.utils.utils import chunks
from punctuation.parser.punctuation_parser import get_textinfo
import spacy
from multiprocessing import Pool, cpu_count
#from  threading import Thread 

import timeout_decorator

spacy.load('en_core_web_sm')
logger = logging.getLogger(__name__)


#@logging_function(logger)
def ranks_of_freq(freq):
    return np.array(freq).argsort()[::-1].tolist()

#@logging_function(logger)
@timeout_decorator.timeout(40, use_signals=False)
def get_gutenberg_text(book_id):
    """
    This function gets the text corresponding to the book_id 
    from Gutenberg database.
    """
    try:
        x = strip_headers(load_etext(int(book_id), prefer_ascii=False)).strip()
    except:
        x = None
    return x

def get_gutenberg_texts(book_ids):
    """
    This function gets the texts corresponding to the list of book_ids
    from Gutenberg database.
    """
    list_texts = []
    for book_id in book_ids:
        list_texts.append(get_gutenberg_text(book_id))
    return list_texts

def get_gutenberg_texts_tokens(list_book_ids):
    """
    This function gets the texts corresponding to the list of book_ids
    from Gutenberg database.
    """
    list_texts = []
    list_tokens = []
    for book_id in list_book_ids:
        try:
            text = get_gutenberg_text(book_id)
        except:
            text = None
            print('Timed out. Could not find: {}'.format(book_id))
        list_texts.append(text)
        tokens = get_textinfo(text)
        list_tokens.append(tokens)
    return list_texts, list_tokens

#@logging_function(logger)
def get_gutenberg_texts_pool(list_book_ids):
    total_threads = cpu_count()
    chunk_size = int(len(list_book_ids) / total_threads) + 1
    sets_to_be_computed = chunks(list_book_ids, chunk_size)
    pool = Pool(total_threads)
    results = pool.map(get_gutenberg_texts, sets_to_be_computed)
#    results = Thread(get_gutenberg_texts, sets_to_be_computed)
    l = []
    for l_res in results:
        l = l+l_res
    pool.close()
    pool.join()
    return l

def get_gutenberg_text_tokens_pool(list_book_ids, save_text=False):
    total_threads = cpu_count()
    chunk_size = int(len(list_book_ids) / total_threads) + 1
    sets_to_be_computed = chunks(list_book_ids, chunk_size)
    pool = Pool(total_threads)
    results = pool.map(get_gutenberg_texts_tokens, sets_to_be_computed)
#    results = Thread(get_gutenberg_texts_tokens, sets_to_be_computed)
    l_tokens = []
    l_text = []
    for l_res in results:
        if save_text: l_text = l_text+l_res[0]
        l_tokens = l_tokens+l_res[1]
    pool.close()
    pool.join()
    return l_text, l_tokens