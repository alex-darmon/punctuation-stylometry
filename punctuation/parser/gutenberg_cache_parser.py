#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:26:32 2019

@author: alexandradarmon
"""

import numpy as np
import pandas as pd
import gutenberg.acquire
import logging
from logs.logger import logging_function
from punctuation.utils.utils import splitter_function
logger = logging.getLogger(__name__)

from gutenberg.query import list_supported_metadatas
print(list_supported_metadatas())


@logging_function(logger)
def get_max_book_id():
    max_book_id = 1000
    return max_book_id

@logging_function(logger)
def get_min_book_id():
    min_book_id = 0
    return min_book_id

@logging_function(logger)
def get_list_book_id():
    list_book_id = range(0,1000)
    return list_book_id


@logging_function(logger)
def random_book_ids(n , list_n=None):
    if list_n is None:
        list_n = get_list_book_id()
    return np.random.choice(list_n, n).tolist()


@logging_function(logger)
def random_book_id(list_n=None):
    if list_n is None:
        list_n = get_list_book_id()
    return np.random.choice(list_n,1).tolist()[0]


def get_cache_info(list_epubs,
                   verbose=True,
                   cache_data_directory='data/cache/epub'):
    titles = []
    authors = []
    author_birthdates = []
    author_deathdates = []
    languages = []
    genres = []
    subjects = []
    book_ids = []
    count = 0
    for directory_nb in list_epubs:
        
        if count%100==0 and verbose : print(count)
        count+=1
        
        book_ids.append(directory_nb)
        file_name = cache_data_directory+'/'+str(directory_nb)+'/pg'+str(directory_nb)+'.rdf'
        
        try:
            data = open(file_name, 'r').read()
            title = splitter_function(data, '<dcterms:title>','</dcterms:title>')
            titles.append(title)
            
            book_shelf = splitter_function(data, '<pgterms:bookshelf>', '</pgterms:bookshelf>')
            genre = splitter_function(book_shelf, '<rdf:value>', '</rdf:value>')
            genres.append(genre)
            
            res_subjects = []
            if '<dcterms:subject>' in data:
                subject_sections = data.split('<dcterms:subject>')
                for subject_section in subject_sections[1:]:        
                    subject = splitter_function(subject_section, '<rdf:value>', '</rdf:value>')
                    res_subjects.append(subject)
            subjects.append(res_subjects)
            
            author_section = splitter_function(data, '<dcterms:creator>', '</dcterms:creator>')
            author = splitter_function(author_section, '<pgterms:name>','</pgterms:name>')
            authors.append(author)
            
            bithdate_section = splitter_function(author_section,
                                                 '<pgterms:birthdate',
                                                 '</pgterms:birthdate>')
            if bithdate_section is not None:
                bithdate = bithdate_section.split('>')[-1]
            else:
                bithdate = None
            author_birthdates.append(bithdate)
                
            deathdate_section = splitter_function(author_section, 
                                                  '<pgterms:deathdate',
                                                  '</pgterms:deathdate>')
            if deathdate_section is not None:
                deathdate = deathdate_section.split('>')[-1]
            else:
                deathdate = None
            author_deathdates.append(deathdate)
                
            language_section = splitter_function(data, '<dcterms:language>', '</dcterms:language>')
            language = splitter_function(language_section,
                                         '<rdf:value rdf:datatype="http://purl.org/dc/terms/RFC4646">',
                                         '</rdf:value>')
            
            languages.append(language)
        except:
            titles.append(None)
            authors.append(None)
            author_birthdates.append(None)
            author_deathdates.append(None)
            genres.append(None)
            subjects.append(None)
            languages.append(None)
    
    df_res = pd.DataFrame()
    df_res['title'] = titles
    df_res['author'] = authors
    df_res['author_birthdate'] = author_birthdates
    df_res['author_deathdate'] = author_deathdates
    
    df_res['genre'] = genres
    len(df_res['genre'].dropna())
    #16340
    len(df_res['genre'])
    #56645
    
    df_res['language'] = languages
    df_res['subject'] = subjects
    df_res['book_id'] = book_ids
    return df_res