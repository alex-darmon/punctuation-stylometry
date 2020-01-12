#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:46:01 2019

@author: alexandradarmon
"""

import pandas as pd
import numpy as np
from punctuation.config import options
from webcolors import hex_to_rgb
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from PIL import ImageDraw
import math as ma

color_vector = ['#2764A2','#EC7428','#438823', '#B9312B', '#785BAD','#72473D',
                '#CA6FB6', '#6C6C6C','#B1AC27', '#44ADC1']
rgb_color_vector = [hex_to_rgb(i) for i in color_vector]




def plot_punc_color(title, author, df=df):
   
    
    
    
    ## VARIABLES
    # size of output canvas in pixels
    canvasHeight = 800;
    canvasWidth = 800;
    
    # pixel border width
    trim = 100;
    
    # number of symbols to be output on each line
    symbolsPerLine = 70;
    # and the number of lines
    linesOfText = 70;
    
    # symbolsPerLine = int(math.floor(math.sqrt(len(punct))));
    # linesOfText = int(math.floor(len(punct)/symbolsPerLine));
    
    
    deltaW = (canvasWidth - trim*2)/symbolsPerLine
    deltaH = (canvasHeight - trim*2)/linesOfText
    
    bkgColor = (255,255,255)
    
    transitionFill = (0,0,0);
    endSentenceFill = (0,0,0);
    parentheticalFill = (0,0,0);

    
    punct = df[df.title==title]['seq_pun'].iloc[0]
    punct = punct[int(len(punct)/2):]
    linesOfText = symbolsPerLine = int(min(ma.sqrt(3000), ma.sqrt(len(punct))))
    img = Image.new("RGB", [canvasWidth,canvasHeight], bkgColor)
    draw = ImageDraw.Draw(img)
    
    for ii in range(linesOfText):
       for jj in range(symbolsPerLine):
          symb = punct[jj + ii*symbolsPerLine]
          if (symb == '.'):
             draw.text((trim + jj*deltaW,trim + ii*deltaH ), symb,fill=endSentenceFill)
          elif (symb == ','):
             draw.text((trim + jj*deltaW,trim + ii*deltaH ), symb,fill=transitionFill)
          elif (symb == '!') or (symb == '?'):
             draw.text((trim + jj*deltaW,trim + ii*deltaH), symb,fill=endSentenceFill)
          elif (symb == '"') or (symb == '\'') or (symb == '(') or (symb == ')') or (symb == '[') or (symb == ']'):
             draw.text((trim + jj*deltaW,trim + ii*deltaH), symb,fill=parentheticalFill)
          elif (symb == ';') or (symb == '-') or (symb == ':'):
             draw.text((trim + jj*deltaW,trim + ii*deltaH), symb,fill=transitionFill)
          else:
             draw.text((trim + jj*deltaW,trim + ii*deltaH), symb,fill=transitionFill)
    
    img.save('figures/raw_data/'+author+'/'+title.replace(' ','_') + 'raw_seq.eps')
    
    print(len(punct))

def plot_punc_heatmap(title, author=None, df=None):
    
    punct = df[df.title==title]['seq_pun'].iloc[0]
    punct = punct[int(len(punct)/2):]
    
    linesOfText = symbolsPerLine = int(min(ma.sqrt(3000), ma.sqrt(len(punct))))
    linesOfText = symbolsPerLine = int(min(ma.sqrt(3000), ma.sqrt(len(punct))))
    array = np.zeros([linesOfText, symbolsPerLine, 3], dtype=np.uint8)
    
    for ii in range(linesOfText):
       for jj in range(symbolsPerLine):
           symb = punct[jj + ii*symbolsPerLine]
           array[ii,jj,:] = rgb_color_vector[options.punctuation_vector.index(symb)]
    img = Image.fromarray(array)
    img.show()
    if author is not None:
        img.save('figures/raw_data/'+author+'/'+title.replace(' ','_')+'_heatmap.png')
    else:
        img.save('figures/raw_data/other/'+title.replace(' ','_')+'.png')


def plot_legend_raw_data():
    fig=plt.figure(figsize=(8,1))
    ax=fig.add_subplot(111)
    vals= list(range(0,11))
    cmap = mpl.colors.ListedColormap(['#2764A2','#EC7428','#438823',
                                      '#B9312B', '#785BAD','#72473D',
                                      '#CA6FB6', '#6C6C6C','#B1AC27',
                                      '#44ADC1'])
    norm = mpl.colors.BoundaryNorm(vals, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    spacing='uniform',
                                    orientation='horizontal',
                                    extend='neither',
                                    ticks=vals)
    cb.ax.set_xticklabels(options.punctuation_vector[:-1]+['...'],horizontalalignment='center',)
    cb.ax.tick_params(labelsize=30, )
