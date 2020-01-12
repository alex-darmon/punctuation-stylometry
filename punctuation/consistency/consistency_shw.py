#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:01:24 2019

@author: alexandradarmon
"""



plot_consitency(feature_name, df_consistency_within, 
                baseline_between, col_name='author',
                with_legend=False, 
                path_consistency=None,
                to_show=False)

kl_norm_trans_auth = class_consistency(df_consistency_within, feature_name,
                                       col_name=col_name)
kl_norm_trans_auth.sort_values('normalised_tran_mat_compare', ascending=False, inplace=True)
list_authors = kl_norm_trans_auth.author.tolist()

sh = 'Shakespeare, William'
wells = 'Wells, H. G. (Herbert George)'

rk_sh = list_authors.index(sh)
rk_w = list_authors.index(wells)

plt.axvline(rk_sh, color='yellow', linestyle='-')
plt.axvline(rk_w, color='magenta', linestyle='-')
plt.savefig(path_consistency+'/plot_consistency_{}.png'.format(feature_name))
plt.show()
