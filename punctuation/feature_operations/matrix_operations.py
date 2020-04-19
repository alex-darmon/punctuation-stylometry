#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:36:50 2019

@author: alexandradarmon
"""

import numpy as np
from punctuation.config import options

def update_mat_tot(tot,mat,char1,char2):
    ind1 = options.punctuation_vector.index(char1)
    ind2 = options.punctuation_vector.index(char2)
    s = tot[ind1]= tot[ind1]+1
    mat[ind1,:] = mat[ind1,:]*(s-1)
    mat[ind1,ind2] = mat[ind1,ind2]+1
    mat[ind1,:] = mat[ind1,:]/s
    
    
#This function gives the transition matrix of a 
#sequence of punctuation marks
def transition_mat(seq_pun):
    try:
        transition_mat =np.zeros((len(options.punctuation_vector),
                                  len(options.punctuation_vector)), dtype='f')
        count_pun =  np.zeros(len(options.punctuation_vector), dtype='f')
    
        for i in range(0,len(seq_pun)-1):
            update_mat_tot(count_pun,
                           transition_mat,seq_pun[i],seq_pun[i+1])
        return transition_mat
    except:
        return None

# normalized with frequencies
def normalised_transition_mat(mat,freq_pun):
    try:
        res = mat.copy()
        for i in range(0,len(freq_pun)):
            res[i,:] = res[i,:]*freq_pun[i]
        return res
    except:
        return None
    
# normalized with frequencies
def get_transition_mat(norm_mat,freq_pun):
    try:
        res = norm_mat.copy().reshape((10,10))
        for i in range(0,len(freq_pun)):
            if freq_pun[i]>0:
                res[i,:] = res[i,:]/freq_pun[i]
        return res
    except:
        return None

def update_mat_nb_words(tot,mat,char1,char2,count):
    ind1 = options.punctuation_vector.index(char1)
    ind2 = options.punctuation_vector.index(char2)
    s = tot[ind1,ind2] = tot[ind1,ind2]+1
    mat[ind1,ind2] = (mat[ind1,ind2]*(s-1)+count)/s

#function to compute a matrix with the mean of
# words between two punctuation marks:
def mat_nb_words_pun(seq_nb_words_pun):
    try:
        mat_nb_word_pun = np.zeros((len(options.punctuation_vector), 
                                    len(options.punctuation_vector)), dtype='f')
        count_pun =  np.zeros((len(options.punctuation_vector),
                               len(options.punctuation_vector)), dtype='f')
        for i in range(1,len(seq_nb_words_pun)-2, 2):
            update_mat_nb_words(count_pun, mat_nb_word_pun,
                seq_nb_words_pun[i], seq_nb_words_pun[i+2],
                seq_nb_words_pun[i+1])
        
        return mat_nb_word_pun
    except:
        return None