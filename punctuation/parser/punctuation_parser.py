#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 03:36:59 2019

@author: alexandradarmon
"""

import pandas as pd
import logging
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
from punctuation.config import options
from punctuation.feature_operations.matrix_operations import (
        normalised_transition_mat,
        transition_mat,
        mat_nb_words_pun
)
logger = logging.getLogger(__name__)



#@logging_function(logger)
def get_tokens_word_nb_punctuation(
        tokens, 
        punctuation_vector=options.punctuation_vector,
        punctuation_quotes=options.punctuation_quotes,
        alpha=options.alpha
        ):
    """
    This function returns a list of punctuation marks with the number of 
    words in between when given list of tokens from spacy.
    """
    try:
        tokens_word_nb_punctuation = []
        count = 0
        for token in tokens:
            token = str(token)
            if token in punctuation_vector+punctuation_quotes:
                if token in punctuation_quotes:
                    token = '"'
                tokens_word_nb_punctuation += [count, token]
                count = 0
            elif len(set(alpha).intersection(set(str(token))))>0:
                count+=1
        return tokens_word_nb_punctuation
    except:
        return None


# @logging_function(logger)
def get_textinfo(text):
    if text is None:
        return None
    parser = English(max_length=len(text)+1)
    tokens = parser(text)
    raw_seq_nb_words = get_tokens_word_nb_punctuation(tokens)
    return raw_seq_nb_words


#@logging_function(logger)
def get_textinfos(list_book_ids):
    list_raw_seq_nb_words = []
    error_book_id = []
    count = 0
    for book_id in list_book_ids:
        if count%50==0: 
            log_msg = "number of documents processed: %d/%d".\
                format(count, len(list_book_ids))
            logger.debug(log_msg)
        count+=1
        try:
            raw_seq_nb_words = get_textinfo(book_id)
            list_raw_seq_nb_words.append(raw_seq_nb_words)
        except:
            error_book_id.append(int(book_id))
            list_raw_seq_nb_words.append(None)
    
    return list_raw_seq_nb_words


#@logging_function(logger)
def get_frequencies(tokens, vector=options.punctuation_vector):
    try:
        freqs = []
        for elt in vector:
            freqs.append((tokens.count(elt)))
        return list(map(lambda x: x/ sum(freqs), freqs))
    except:
        return None

def seq_nb_only(seq):
    try:
        if type(seq[0])==int: res= seq[0:-1:2]
        else: res = seq[1::2]
        return res
    except:
        return None

def seq_pun_only(seq): 
    try:
        if type(seq[0])==int: res= seq[1::2]
        else: res = seq[0:-1:2]
        return res
    except:
        return None


def get_tokens_sentences_nb(tokens,
                            punctuation_vector=options.punctuation_vector,
                            punctuation_end=options.punctuation_end,
                            include_pun=False):
    try:
        tokens_sentences_nb = []
        count = 0
        for token in tokens:
            if token in punctuation_end:
                tokens_sentences_nb+= [count, token]
                count = 0
            else:
                if token not in punctuation_vector:
                    if type(token) == int:
                        count+=token
        return tokens_sentences_nb
    except:
        return None
    
    

def enrich_features(df, text_col='text', inplace=True):
    if inplace:
        df['tokens'] = df['text'].apply(get_textinfo)
        
        df['seq_pun'] =  df['tokens'].apply(seq_pun_only)
        df['freq_pun'] = df['seq_pun'].apply( lambda x:
            get_frequencies(x, options.punctuation_vector))
            
        df['seq_nb_only'] = df['tokens'].apply(seq_nb_only)
        df['freq_word_nb_punctuation'] = df['seq_nb_only'].\
                apply(lambda x: get_frequencies(x, range(0, options.empirical_nb_words)))
        
        df['seq_length_sen'] = df['tokens'].apply(get_tokens_sentences_nb)
        df['freq_length_sen'] = df['seq_length_sen'].apply(
                lambda x: get_frequencies(x, range(0, options.empirical_nb_sentences)))
        
        df['tran_mat'] = df['seq_pun'].apply(transition_mat)
        df['normalised_tran_mat'] = df.apply(lambda row: 
                normalised_transition_mat(row['tran_mat'], row['freq_pun']), axis=1)
        df['mat_nb_words'] = df['tokens'].apply(mat_nb_words_pun)
        
        df['tran_mat'] = \
            df['tran_mat'].apply(lambda x:x.flatten())
        df['normalised_tran_mat'] = \
            df['normalised_tran_mat'].apply(lambda x:x.flatten())
        df['mat_nb_words'] = \
            df['mat_nb_words'].apply(lambda x:x.flatten())
    
        df[options.freq_pun_col] = \
            pd.DataFrame(df.freq_pun.values.tolist(),
                         index=df.index)
        
        df[options.freq_nb_words_col] = \
            pd.DataFrame(df.freq_word_nb_punctuation.values.tolist(),
                         index=df.index)
            
        df[options.freq_length_sen_with_col] = \
            pd.DataFrame(df.freq_length_sen.values.tolist(),
                         index=df.index)
        
        df[options.transition_mat_col] = \
            pd.DataFrame(df.tran_mat.values.tolist(),
                         index=df.index)
            
        df[options.norm_transition_mat_col] = \
            pd.DataFrame(df.normalised_tran_mat.values.tolist(),
                         index=df.index)
        
        df[options.mat_nb_words_pun_col] = \
            pd.DataFrame(df.mat_nb_words.values.tolist(),
                         index=df.index)
            
    else:
        df_bis = df.copy()
        enrich_features(df_bis, text_col=text_col, inplace=True)
        return df_bis