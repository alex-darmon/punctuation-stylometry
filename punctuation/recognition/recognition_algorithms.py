#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:10:22 2019

@author: alexandradarmon
"""

import numpy as np
from sklearn.metrics import accuracy_score
from punctuation.utils.utils import (
        try_to_load_as_pickled_object_or_None, 
        load_corpus,
        save_as_pickled_object)
import pandas as pd
from punctuation.config import options

from punctuation.feature_operations.distances import d_KL
from sklearn.neural_network import MLPClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import  StandardScaler


def get_acc_kl_nearest_neigh(index_train, index_test, feature,
                    feature_name, df=None,
                    distance=d_KL, col='author', #save=True,
                    path_nn = 'data/pickle/recognition/author'):
    
    if df is None: df = load_corpus()
    
    X = df[[feature_name]]
    y = df[[col]]

    X_train, X_test = X.loc[index_train], X.loc[index_test]
    y_train, y_test = y.loc[index_train], y.loc[index_test]
    X_train[col] = y_train[col].tolist()
    
    nb_classes = len(X_train[[col]].drop_duplicates())
    df_compare = try_to_load_as_pickled_object_or_None('{}/nearest_neigh_{}_{}.p'.format(path_nn,
                                  nb_classes, feature_name))
    df_signature = X_train[[col]+[feature_name]].\
            groupby(col)[feature_name].\
            apply(lambda x: np.mean(np.array(list(x)),axis=0))
    df_signature = df_signature.reset_index()
   # df_res = df_signature[[col]].reset_index()
    df_signature['to_join'] = 1

    if df_compare is None:        
        
        df_res = X_test[[feature_name]]
        df_res['index_test'] = df_res.index
        df_res['to_join'] = 1
    
        df_compare = pd.merge(df_signature, df_res, on='to_join',
                              suffixes=('_signature','_candidate'))
        
        df_compare['distance'] = df_compare.\
            apply(lambda row: distance(row[feature_name+'_signature'],
                                       row[feature_name+'_candidate']), axis=1)
        
    df_res = df_compare.\
        iloc[df_compare.groupby('index_test')['distance'].\
             idxmin()][['index_test',col]]
    
    df_res['index_cat'] = pd.Categorical(
    df_res['index_test'], 
    categories=index_test, 
    ordered=True)
    df_res.sort_values('index_cat',inplace=True)
    
    predictions = df_res[col]
    acc = accuracy_score(y_test, predictions)
    print(acc)
    print(str((feature_name, nb_classes, len(index_train), 
               len(index_test),  acc)),
              file=open('data/pickle/recognition/author/res_neares_recognistion.txt','a'))
    
#    if save:
    save_as_pickled_object(df_compare, '{}/nearest_neigh_{}_{}.p'.format(path_nn,
                                      nb_classes, feature_name))

    return (nb_classes, len(X_train), len(X_test),
            accuracy_score(y_test, predictions))



def get_acc_neural_net(index_train, 
                index_test,
                feature_name,
                features, df,
                distance, col,
                layers=2000,
#                save=True,
                path_nn='data/pickle/recognition/author'):    
                    
    X = df[features]
    y = df[[col]]
    
    X_train, X_test = X.loc[index_train], X.loc[index_test]
    y_train, y_test = y.loc[index_train], y.loc[index_test]
    
    nb_classes = len(y_train[[col]].drop_duplicates())
    
    df_compare = try_to_load_as_pickled_object_or_None('{}/neural_net_{}_{}.p'.format(path_nn,
                                          nb_classes, feature_name))
    if df_compare is None:
 
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        
        mlp = MLPClassifier(hidden_layer_sizes=layers,
                            alpha = 0.000000001,
                            tol=0.000001,
                            max_iter=1000,
                            learning_rate= 'adaptive',
                            verbose=True,
                            )
        mlp.fit(X_train,y_train)
        predictions = mlp.predict(X_test)
        predictions_proba = mlp.predict_proba(X_test)
    
        df_compare = pd.DataFrame.from_dict({'predictions':predictions,
                                   'y_test': y_test,
                                   'predictions_proba':predictions_proba.tolist()},
                            orient='columns')

    predictions = df_compare['predictions'].tolist()
    
    acc = accuracy_score(y_test, predictions)
#    print(acc)
#    print(str((feature_name, nb_classes, len(index_train), 
#               len(index_test),  acc)),
#              file=open('data/pickle/recognition/author/res_neuralnet_recognition.txt','a'))
    
#    if save:
    save_as_pickled_object(df_compare, '{}/neural_net_{}_{}.p'.format(path_nn,
                                      nb_classes, feature_name))

    return (nb_classes, len(X_train), len(X_test),
            acc)



def apply_acc_reco(s, fun):
    (index_train, index_test,
     feature_name, features, df,
     distance, col, path_nn) = s
    (nb_classes, size_train, size_test,
     acc) = fun(index_train, 
                index_test,
                feature_name, features, df,
                distance, col, path_nn=path_nn)
    return (nb_classes, feature_name, size_train, size_test, acc)


def launch_nearest_neighbour(df, df_index_nn, 
                             list_features=options.feature_names,
                             col_class='author',
                             path_index_nn = 'data/pickle/recognition/author',
                             path_res='results/recognition/author'):

    # Recognition Nearest Neighbours
    set_to_compute = []
    for i in range(0, len(df_index_nn)):
        (nb_classes,
         index_train, 
         index_test) = df_index_nn.iloc[i]
        for feature in list_features:
            set_to_compute.append((index_train, index_test,
                                   feature, feature, df,
                                     d_KL, col_class, path_index_nn))
    
    list_r = []
    for s in set_to_compute:
        r = apply_acc_reco(s, get_acc_kl_nearest_neigh)
        list_r.append(r)
    df_res = pd.DataFrame(list_r, columns=['nb_classes',
                                           'feature_name', 
                                           'train size', 'test size',
                                           'accuracy'])
    df_res.to_csv('{}/{}_nearest_neihbour_accuracy.csv'.format(path_res,
                  col_class))
    return df_res


def launch_neural_net(df, df_index_nn,list_features,
                      feature_names, distance=d_KL,
                      col_class='author',
                      path_index_nn = 'data/pickle/recognition/author',
                      path_res='results/recognition/author'):
    
    # Recognition  Neural Networks  
    set_to_compute_neural_net = []
    for i in range(0, len(df_index_nn)):
        (nb_classes,
         index_train, 
         index_test) = df_index_nn.iloc[i]
        for features,feature_name in zip(list_features,feature_names):
            set_to_compute_neural_net.append((index_train, index_test,feature_name,
                                     features, df,
                                     distance, col_class, path_index_nn))

    list_r = []
    for s in set_to_compute_neural_net:
        r = apply_acc_reco(s, get_acc_neural_net)
        list_r.append(r)
    df_res = pd.DataFrame(list_r, columns=['nb_classes',
                                           'feature_name',
                                           'train size', 'test size',
                                           'accuracy'])
    df_res.to_csv('{}/{}_neural_net_accuracy.csv'.format(path_res, col_class))
    return df_res
