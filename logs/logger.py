#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:05:45 2019

@author: alexandradarmon
"""

import logging
from punctuation.config import options
from time import time


def create_logger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("punctuation_logger")
    logger.setLevel(logging.INFO)
    #logger.setLevel(logging.NOTSET) # Set Logger's level to NOTSET, default is WARNING

    # create the logging file handler
    if options.log_file is not None:
        fh = logging.FileHandler(options.log_file)
 
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        fh.setLevel(logging.NOTSET)
        # add handler to logger object
        logger.addHandler(fh)
    return logger


def adjust_arg(arg):
    if type(arg)==str:
        return arg[:75] + '..'
    return arg


def logging_function(logger):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
 
    @param logger: The logging object
    """
 
    def decorator(f):
 
        def wrapper(*args, **kw):
            try:
                ts = time()
                result = f(*args, **kw)
                te = time()
                truncated_args = [adjust_arg(data) for data in args]
                log_msg = 'func:%r args:[%r, %r] took: %2.4f sec' % \
                  (f.__name__, truncated_args, kw, te-ts)
                logger.info(log_msg)
                return result
                
            except:
                # log the exception
                err = "There was an exception in  "
                err += f.__name__
                logger.exception(err)
 
            # re-raise the exception
            raise
        return wrapper
    return decorator

