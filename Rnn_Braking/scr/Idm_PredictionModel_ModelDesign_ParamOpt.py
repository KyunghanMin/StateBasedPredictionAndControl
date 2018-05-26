# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:08:37 2018

@author: Kyunghan
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
#%% Import result
# From: Idm_PredictionModel_ParamOpt.py
with open('HyperOpt_History.pickle','rb') as myloaddata:
    HyperOptResult = pickle.load(myloaddata)
#%% Plot parameter optimization results
History = HyperOptResult[0]
History_Score = []
History_ScoreMin = [History[0]['val_loss'][-1]]
for key in History.keys():
    History_Score.append(History[key]['val_loss'][-1])
    if History_Score[-1] <= History_ScoreMin[-1]:
       History_ScoreMin.append(History_Score[-1])
    else:
       History_ScoreMin.append(History_ScoreMin[-1])    
del History_ScoreMin[0]
#%%
History_Score = np.array(History_Score)
History_ScoreMin = np.array(History_ScoreMin)
plt.plot(History_Score,'o',ms=3)
plt.plot(History_ScoreMin,lw=2)
plt.ylim(0,0.007)
plt.xlabel('Iteration [-]')
plt.ylabel('Validation MSE [m/s$^2$]')
#%% Determine best model_structure
ModelStr_Config = HyperOptResult[2]
ModelStr_Best =  HyperOptResult[1]
# Load best config index

KeyListIndex = sorted(ModelStr_Best.keys())
#===================================================================================================#
#['OptPar_BatchSize', 'OptPar_DenseAct', 'OptPar_InputSeq', 'OptPar_Opt', 'OptPar_SeqMod', 'OptPar_StateNum']
#===================================================================================================#
KeyListConfig = sorted(ModelStr_Config.keys())
#===================================================================================================#
# ['BatchSizeRange', 'DenseActiveSet', 'InputSeqRange', 'Optimizer', 'SeqModelSet', 'StateNumRange']
#===================================================================================================#
ConfigIndex = []
for key in KeyListIndex:
    ConfigIndex.append(ModelStr_Best[key])    

tmpNum = 0    
ModelStr = {}
for key in KeyListConfig:    
    tmpConfigVal = ModelStr_Config[key]
    ModelStr[key] = tmpConfigVal[ConfigIndex[tmpNum]]
    tmpNum = tmpNum+1