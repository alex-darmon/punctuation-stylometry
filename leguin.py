#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:38:06 2020

@author: alexandradarmon
"""

from punctuation.config import options
from punctuation.parser.punctuation_parser import (
 get_textinfo,
 get_frequencies,
 )

text = """
I don’t have a gun and I don’t have even one wife and my sentences tend to go on
 and on and on, with all this syntax in them. Ernest Hemingway would have died 
 rather than have syntax. Or semicolons. I use a whole lot of half-assed semicolons; 
 there was one of them just now; that was a semicolon after "semicolons," 
 and another one after "now."
 """
 
tokens = get_textinfo(text)
# [25, ',', 6, '.', 9, '.', 2, '.', 9, ';', 7, ';', 5, '"', 1, ',', 0, '"', 4, '"', 1, '.', 0, '"']



vector=options.punctuation_vector
freqs = []
for elt in vector:
    freqs.append((tokens.count(elt)))
get_frequencies(tokens, vector)

['!', '"', '(', ')', ',', '.', ':', ';', '?', '^']
[0, 4/12, 0, 0, 2/12, 4/12, 0, 2/12, 0, 0]
#  , . . . ; ; " , " " . "

# , . 1/2  * 2/12
# , " 1/2  * 2/12

# . . 2/4 * 4/12
# . ; 1/4 * 4/12
# . " 1/4 * 4/12


# " . 1/3 * 4/12 = 1/9
# " , 1/3 * 4/12 = 1/9
# " " 1/3 * 4/12 = 1/9


# ; ; 1/2  * 2/12
# ; " 1/2  * 2/12



