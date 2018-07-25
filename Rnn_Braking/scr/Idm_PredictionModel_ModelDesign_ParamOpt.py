# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:08:37 2018

@author: Kyunghan
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import lib_sbpac
#%% Import result
# From: Idm_PredictionModel_ParamOpt.py
with open('HyperOpt_History.pickle','rb') as myloaddata:
    HyperOptResult = pickle.load(myloaddata)
    HyperOptParamSet = pickle.load(myloaddata)
    History = pickle.load(myloaddata)
#%% Plot parameter optimization results
History_Score = []
History_ScoreMin = [History[0]['val_loss'][-1]]
MinimumIndex = []
MinimumCri = 0.0006
for key in range(len(History)):
    History_Score.append(History[key]['val_loss'][-1])
    if History_Score[-1] <= History_ScoreMin[-1]:
        History_ScoreMin.append(History_Score[-1])
    else:
        History_ScoreMin.append(History_ScoreMin[-1]) 
    if History_Score[-1] <= MinimumCri:
        MinimumIndex.append(key)
del History_ScoreMin[0]
#%%
plt.close('all')
Color = lib_sbpac.get_ColorSet()
History_Score = np.array(History_Score)
History_ScoreMin = np.array(History_ScoreMin)
plt.plot(History_Score,'o', ms=3, label = 'TPE score', color = Color['BP'][5], mfc = Color['BP'][8])
plt.plot(History_ScoreMin, lw=2, label = 'Min score value', c = Color['RP'][3])
plt.ylim(0,0.007)
plt.xlabel('Iteration [-]')
plt.ylabel('Validation MSE [m/s$^2$]')
plt.legend()
#%% Determine best model_structure
ModelStr_Best =  HyperOptResult
# Load best config index

KeyListIndex = sorted(ModelStr_Best.keys())
#===================================================================================================#
#['OptPar_BatchSize', 'OptPar_DenseAct', 'OptPar_InputSeq', 'OptPar_Opt', 'OptPar_SeqMod', 'OptPar_StateNum']
#===================================================================================================#
OptParam = {'OptPar_InputSeq':range(6,16,2)}
OptParam['OptPar_StateNum'] = range(5,15,1)
OptParam['OptPar_SeqAct'] = ['relu','softmax','sigmoid','tanh','hard_sigmoid','linear']
OptParam['OptPar_RecSeqAct'] = ['relu','softmax','sigmoid','tanh','hard_sigmoid','linear']
OptParam['OptPar_DenseAct'] = ['relu','softmax','sigmoid','tanh','hard_sigmoid','linear']
OptParam['OptPar_BatchSize'] = range(10,100,20)
OptParam['OptPar_Opt'] = ['SGD','Adagrad','RMSprop','Adam','Nadam']
KeyListConfig = sorted(OptParam.keys())
#===================================================================================================#
# ['BatchSizeRange', 'DenseActiveSet', 'InputSeqRange', 'Optimizer', 'SeqModelSet', 'StateNumRange']
#===================================================================================================#
ConfigIndex = []
for key in KeyListIndex:
    ConfigIndex.append(ModelStr_Best[key])    

tmpNum = 0    
ModelStr = {}
for key in KeyListConfig:    
    tmpConfigVal = OptParam[key]
    ModelStr[key] = tmpConfigVal[ConfigIndex[tmpNum]]
    tmpNum = tmpNum+1