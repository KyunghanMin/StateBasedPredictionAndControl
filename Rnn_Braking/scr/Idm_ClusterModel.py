# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:43:53 2018

@author: Kyunghan
Description: Cluster algorithm development using prediction results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

#%% Load prediction result
PredictData = io.loadmat('PredictResult.mat')
tmpData = PredictData['Data'][0,:]
Driver = tmpData[0]
PredictResult = tmpData[1]
PredictState = tmpData[2]
PredictSequence = tmpData[3]