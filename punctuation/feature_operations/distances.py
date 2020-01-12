#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:29:56 2018

@author: alexandra.darmon
"""

import numpy as np
import math as ma
from scipy.stats import pearsonr


def pearson(x,y):
    return pearsonr(x,y)[0]

def fit_freq_mod2(freq1, freq2):
    if len(freq1)==len(freq2): 
        n = len(freq1)
    else: 
        raise Exception ("Vector of different size")
    new_freq1 = [freq1[i] for i in range(0,n)]
    new_freq2 = [freq2[i] for i in range(0,n)]

    for j in range(0,n):
        if (new_freq1[j] == 0.0):
            q = new_freq2[j]
            new_freq2 = list(map(lambda x: x/(1.0-q),new_freq2))
            new_freq2[j] = 0.0
            
        if (new_freq2[j] == 0.0):
            q = new_freq1[j]
            new_freq1 = list(map(lambda x: x/(1.0-q),new_freq1))
            new_freq1[j] = 0.0
            
    return (new_freq1, new_freq2)
    

def d_KL(freq1, freq2):
    res = 0
    try:
        freq1, freq2 = fit_freq_mod2(freq1, freq2)
    except:
        return 0
    if freq1 is not None and freq2 is not None:
        for i in range(0, len(freq1)): 
            p = freq1[i]
            q = freq2[i]
            if(p*q != 0): res += p*ma.log(p/q)
    return res


def d_l2(freq1, freq2):
    res = 0
    if freq1 is not None and freq2 is not None:
        for i in range(0, len(freq1)): 
            p = freq1[i]
            q = freq2[i]
            res += (p-q)**2
    return ma.sqrt(res)


#def d_Yang(freq1, freq2):
#    #(new_freq1, new_freq2) = fit_freq_mod2(freq1,freq2)
#    (new_freq1, new_freq2) = (freq1, freq2)
#    rank1 = ranks_of_freq(new_freq1)
#    rank2 = ranks_of_freq(new_freq2)
#    res = 0
#    for i in range(0,max(len(new_freq1),len(new_freq2))):
#        p1 = new_freq1[i]
#        p2 = new_freq2[i]
#        if (rank1[0,i] != None) & (rank2[0,i] != None)\
#            &(p1 != 0) & (p2 != 0):
#            f = (-p1*ma.log(p1) - p2*ma.log(p2))
#            res += float(abs(rank1[0,i] - rank2[0,i]))*f
#    return res

    
def Shannon_entropy(freq):
    res = 0
    for i in range(0,len(freq)): 
        p = freq[i]
        if(p != 0): res += - p*ma.log(p)
    return res

def d_KLD(freq1,freq2):
    res=0
    #(new_freq1,new_freq2) = fit_freq_mod2(freq1,freq2)
    (new_freq1,new_freq2) = (freq1,freq2)
    for i in range(0,len(new_freq1)): 
        p = new_freq1[i]
        q = new_freq2[i]
        if(p*q != 0): res += 1/2.0 *( p*ma.log(2*p/(p+q)) + q*ma.log(2*q/(p+q)))
    return res

def d_KL_mat(mat1,mat2):
    res=0
    pun_vector = ['!', '"', '(', ')', ',', '.', ':', ';', '?', '^']
    for i in range(0,len(pun_vector)):
        for j in range(0,len(pun_vector)):
            pij = mat1[i,j]
            qij = mat2[i,j]
            if(pij*qij != 0): res += pij*ma.log(pij/(qij))
    return res
    

def d_KLD_mat(mat1,mat2):
    res=0
    pun_vector = ['!', '"', '(', ')', ',', '.', ':', ';', '?', '^']
    for i in range(0,len(pun_vector)):
        for j in range(0,len(pun_vector)):
            pij = mat1[i,j]
            qij = mat2[i,j]
            if(pij*qij != 0): res += 1/2.0 * ( pij*ma.log(2*pij/(qij+pij)) + qij*ma.log(2*qij/(qij+pij)))
    return res


def distance_mat_1(mat1,mat2):
    return  np.linalg.norm(mat1-mat2,ord=1)

def distance_mat_2(mat1,mat2):
    return  np.linalg.norm(mat1-mat2,ord=2)

def distance_mat_fro(mat1,mat2):
    return  np.linalg.norm(mat1-mat2,ord='fro')
    
def distance_mat_minus_1(mat1,mat2):
    return  np.linalg.norm(mat1-mat2,ord=-1)
    
def distance_mat_nuc(mat1,mat2):
    return  np.linalg.norm(mat1-mat2,ord='nuc')    

def distance_nb_abs(res1,res2):
    return  abs(res1 - res2)

def distance_nb_2(res1,res2):
    return  (res1 - res2)**2

def distance_list_norm(res1, res2):
    v1 = np.array(res1)
    v2 = np.array(res2)
    return np.linalg.norm(v1-v2)