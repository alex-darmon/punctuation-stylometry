#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:15:39 2019

@author: alexandradarmon
"""

from punctuation.utils.convert_epub2text import epub2txt
from punctuation.utils.utils import chunks
from punctuation.utils.cache_info import random_book_id
from punctuation.parser.gutenberg_parser import (
        get_gutenberg_text,
        get_tokens_word_nb_punctuation
)
from gutenberg.query import get_metadata
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from spacy.lang.en import English
import spacy
spacy.load('en_core_web_sm')

from os import listdir
import logging
from logs.logger import logging_function
logger = logging.getLogger(__name__)


@logging_function(logger)
def get_tokens(text, parser):
    return parser(text)

book_id = random_book_id()
text = get_gutenberg_text(book_id)
parser = English(max_length=len(text)+1)
tokens = get_tokens(text, parser)
raw_seq_nb_words = get_tokens_word_nb_punctuation(tokens)


## get list of epub
import requests
import urllib

## ebooks
page_sitemap1 = requests.get('https://digilibraries.com/sitemap/sitemap1.xml')
list_urls1 = list(map(lambda x: x.split('</loc>')[0], str(page_sitemap1.content)\
                      .split('<loc>')[1:]))
print(len(list_urls1)) #30004


## authors
page_sitemap2 = requests.get('https://digilibraries.com/sitemap/sitemap2.xml')
list_urls2 = list(map(lambda x: x.split('</loc>')[0], str(page_sitemap2.content)\
                 .split('<loc>')[1:]))
print(len(list_urls2)) #3588
set(list_urls2).intersection(set(list_urls1))


def get_ebook_from_url(url, list_exists=listdir('data/sample/')):
    try:
        content_url = str(requests.get(url).content)
        name_file = content_url.split('download')[1].split('"')[0][1:]
        new_url  = 'https://digilibraries.com/download/'+ name_file
        if not(name_file in list_exists):
            urllib.request.urlretrieve(new_url, 'data/sample/%s.epub'%(name_file))
    except:
        print(url)

def get_ebooks_from_urls(list_urls):
    for url in list_urls:
         get_ebook_from_url(url)


#
#
 ### convert epub
#from epub_conversion.utils import open_book, convert_epub_to_lines
#book = open_book("sample/0a0c8a02ccddcc4e8405fdb33262625f.epub")
#lines = convert_epub_to_lines(book)
#from epub_conversion import Converter
#converter = Converter("sample/")
#converter.convert("sample_text.gz")

#text = epub2txt("sample/ffcd77ac3d01c543d50b6ab1a076f7a9.epub").convert()
