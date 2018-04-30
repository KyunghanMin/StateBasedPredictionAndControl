# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:03:49 2018

@author: Kyunghan
"""
import os
import numpy as np
import lib_sbpac.color_code as colorcode
import pickle
#%%
cdir = os.getcwd()
data_dir = os.chdir('ModelData')

with open('HyperOpt_History.pickle','rb') as f:
    OptResult = pickle.load(f)
    
os.chdir(cdir)
#%%
lstMinLossTrain = []
lstMinLossValid = []
for result in OptResult:    
    lstMinLossTrain.append(np.min(OptResult[result]['loss']))
    lstMinLossValid.append(np.min(OptResult[result]['val_loss']))
    
