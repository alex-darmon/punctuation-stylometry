#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:45:25 2019

@author: alexandradarmon
"""

import configargparse
import os

BASEDIR = os.path.join(os.path.dirname('__file__'))# + os.sep + os.pardir + os.sep)


_DEFAULTS = {
    "config": BASEDIR + "conf/punctuation.ini",
    "log_file": "log/message.log",
    "font_size": 22,
    "log_level": "INFO",
    "empirical_nb_words": 40,
    "empirical_nb_sentences": 200,
    "punctuation_vector": ['!', '"', '(', ')', ',', '.', ':', ';', '?', '^'],
    "punctuation_end": ['!', '?', '.', '^'],
    "alpha": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
    "exception_strings": ["Mr", "Mrs","Dr","Prof","St", "etc"],
    "punctuation_quotes" : ["'","“", "”"],
    "genres_to_include" : ["Science Fiction", "Children's Book Series", "Historical Fiction",
             "Children's Fiction", "Bestsellers, American", "1895-1923",
             "Children's Literature", "US Civil War", "Humor",
             "Best Books Ever Listings", "Western", "Christianity"
             "Detective Fiction", "Fantasy", "World War I", "Philosophy",
             "Adventure", "Christmas", "Children's Picture Books",
             "Harvard Classics", "Movie Books", "School Stories",
             "One Act Plays", "Crime Fiction", "Children's History",
             "Poetry", "Crime Nonfiction", "Animal", "Travel",
             "Biology", "Precursors of Science Fiction", "Art",
             "Classical Antiquity", "Children's Instructional Books",
             "World War II", "Horror"]

}


class Config(object):

    def __init__(self, **kwargs):
        if kwargs is not None:
            self._DEFAULTS = kwargs
#            p = configargparse.ArgParser(
#            auto_env_var_prefix="PUNCTUATION_",
#            ignore_unknown_config_file_keys=True
#            )
#            p.set_defaults(**self._DEFAULTS)
#            options = p.parse_args()
#
#            options.nb_signs = len(options.punctuation_vector)
#            options.freq_pun_col = ['FREQ_'+str(i) for i in range(0, options.nb_signs)]
#            options.freq_nb_words_col = ['FREQ_WORD_'+str(i) for i in range(0, options.empirical_nb_words)]
#            options.freq_length_sen_with_col = ['FREQ_SEN_'+str(i) for i in range(0, options.empirical_nb_sentences)]
#            options.transition_mat_col = ['TRANS_'+str(i) for i in range(0, options.nb_signs*options.nb_signs)]          
#            options.norm_transition_mat_col = ['NORM_TRANS_'+str(i) for i in range(0, options.nb_signs*options.nb_signs)]
#            options.mat_nb_words_pun_col = ['MAT_WORD_'+str(i) for i in range(0, options.nb_signs*options.nb_signs)]
            
#            return options
        
    def parse_arguments(self):
        p = configargparse.ArgParser(
            auto_env_var_prefix="PUNCTUATION_",
            ignore_unknown_config_file_keys=True
        )
        p.set_defaults(**self._DEFAULTS)

        # Global configuration parameters
        p_global = p.add_argument_group(title="Global Settings")
        p_global.add_argument("-c", "--config", help = "Config file with default settings", required=False, is_config_file=True)

        # Logging
        p_logging = p.add_argument_group(title="Reporting and Logging Settings")
        p_logging.add_argument("-l", "--log-file", help = "Log file location", default=None)
        p_logging.add_argument("-L", "--log-level", help = "Log level", default="INFO")
        p_logging.add_argument("-v", "--verbose", help = "Verbose", action="store_true", default=False)
        p_logging.add_argument("-d", "--debug", help = "Debug", action="store_true", default=False)
        p_logging.add_argument("-F", "--font-size", help = "Verbose", action="store_true", default=22)

        # Execution parameters
        p_execution = p.add_argument_group(title="Execution Settings")
        p_execution.add_argument("-w", "--empirical-nb-words", help = "Empirical Number of Words", required=False, default=40, type=int)
        p_execution.add_argument("-s", "--empirical-nb-sentences", help = "Empirical Number of Sentences", required=False, default=200, type=int)
        p_execution.add_argument("-p", "--punctuation-vector", help = "Punctuation Vector", required=False, 
                                 default=['!', '"', '(', ')', ',', '.', ':', ';', '?', '^'], nargs="*", type=str)
        p_execution.add_argument("-e", "--punctuation-end", help = "Punctuation End", required=False, 
                                 default=['!', '?', '.', '^'], nargs="*", type=str)
        p_execution.add_argument("-q", "--punctuation-quotes", help = "Punctuation Quotes", required=False, 
                                 default=["'","“", "”"], nargs="*", type=str)
        p_execution.add_argument("-a", "--alpha", help = "List of characters", required=False, 
                                 default="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", type=str)
        p_execution.add_argument("-x", "--exception-strings", help = "Exception Strings", required=False, 
                                 default=["Mr", "Mrs", "Dr", "Prof", "St", "etc"],  nargs="*", type=str)
        p_execution.add_argument("-g", "--genre-included", help = "Genre Strings", required=False, 
                                 default=["Science Fiction", "Children's Book Series", "Historical Fiction",
                                         "Children's Fiction", "Bestsellers, American", "1895-1923",
                                         "Children's Literature", "US Civil War", "Humor",
                                         "Best Books Ever Listings", "Western", "Christianity"
                                         "Detective Fiction", "Fantasy", "World War I", "Philosophy",
                                         "Adventure", "Christmas", "Children's Picture Books",
                                         "Harvard Classics", "Movie Books", "School Stories",
                                         "One Act Plays", "Crime Fiction", "Children's History",
                                         "Poetry", "Crime Nonfiction", "Animal", "Travel",
                                         "Biology", "Precursors of Science Fiction", "Art",
                                         "Classical Antiquity", "Children's Instructional Books",
                                         "World War II", "Horror"],  nargs="*", type=str)
        
        
        options = p.parse_args()

        options.nb_signs = len(options.punctuation_vector)
        options.freq_pun_col = ['FREQ_'+str(i) for i in range(0, options.nb_signs)]
        options.freq_nb_words_col = ['FREQ_WORD_'+str(i) for i in range(0, options.empirical_nb_words)]
        options.freq_length_sen_with_col = ['FREQ_SEN_'+str(i) for i in range(0, options.empirical_nb_sentences)]
        options.transition_mat_col = ['TRANS_'+str(i) for i in range(0, options.nb_signs*options.nb_signs)]          
        options.norm_transition_mat_col = ['NORM_TRANS_'+str(i) for i in range(0, options.nb_signs*options.nb_signs)]
        options.mat_nb_words_pun_col = ['MAT_WORD_'+str(i) for i in range(0, options.nb_signs*options.nb_signs)]
        
        options.feature_names = ['freq_pun','freq_word_nb_punctuation',
                 'freq_length_sen','normalised_tran_mat']
        options.features = [options.freq_pun_col, options.freq_nb_words_col,
            options.freq_length_sen_with_col, options.norm_transition_mat_col]


        
        return options



config = Config(**_DEFAULTS)
options = config.parse_arguments()
#try:
#    options = config.parse_arguments()
#except:
#    options = config
    