# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:52:38 2018

@author: Acelab
"""
import os
import numpy as np
import math
import scipy.io
import random
import sys 
import matplotlib.pyplot as plt
import shelve
import tensorflow as tf
from pathlib import Path
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Reshape
from keras.layers import SimpleRNN, LSTM
from keras import optimizers
from keras import backend as K
import Idm_Lib
from hyperopt import hp, fmin
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt.mongoexp import MongoTrials
import time

#%%
cdir = os.getcwd()
data_dir = os.chdir('../data')

DataLoad = scipy.io.loadmat('TestUpDataVariRan.mat');
del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
del DataLoad['DataLength'];

os.chdir(cdir)

global Numiteration
Numiteration =0
lstModelHistory = []
lstParamHistory = []
#%% Set hyper parameters
OptParam = {'InputSeqRange':range(6,16,2)}
OptParam['StateNumRange'] = range(5,15,1)
#OptParam['SeqModelSet'] = ['SimpleRnn','LSTM']
OptParam['ActiveSet_seq'] = ['relu','softmax','sigmoid','tanh','hard_sigmoid','linear']
OptParam['ActiveSet_seq_recurrent'] = ['relu','softmax','sigmoid','tanh','hard_sigmoid','linear']
OptParam['ActiveSet_dense'] = ['relu','softmax','sigmoid','tanh','hard_sigmoid','linear']
OptParam['BatchSizeRange'] = range(10,100,20)
OptParam['Optimizer'] = ['SGD','Adagrad','RMSprop','Adam','Nadam']
#OptParam['LearnRate'] = range(1,11,1)
space = {
    'OptPar_InputSeq' : hp.choice('OptPar_InputSeq', OptParam['InputSeqRange']),
    'OptPar_StateNum' : hp.choice('OptPar_StateNum', OptParam['StateNumRange']),
    'OptPar_SeqAct': hp.choice('OptPar_SeqAct', OptParam['ActiveSet_seq']),
    'OptPar_RecSeqAct': hp.choice('OptPar_RecSeqAct', OptParam['ActiveSet_seq_recurrent']),
    'OptPar_DenseAct': hp.choice('OptPar_DenseAct', OptParam['ActiveSet_dense']),                
    'OptPar_BatchSize': hp.choice('OptPar_BatchSize', OptParam['BatchSizeRange']),
    'OptPar_Opt' : hp.choice('OptPar_Opt', OptParam['Optimizer']),
    } 
# params = space 
#%% Determine repeated function
def objective(params):
    # Params = parameters for optimization
    #    --- Input sequence
    #    --- State bumber of sequence model
    #    --- Sequential model 
    #    --- Activation function of dense model
    #    --- Batch size
    #    --- Optimizer of model
    OptParam_InputSequence = params['OptPar_InputSeq']    
    OptParam_NumSeqStateUnit = params['OptPar_StateNum']
    OptParam_SeqActSet = params['OptPar_SeqAct']    
    OptParam_RecSeqActSet = params['OptPar_RecSeqAct']
    OptParam_DenseActSet = params['OptPar_DenseAct']    
    OptParam_BatchSize = params['OptPar_BatchSize']
    OptParam_Optimizer = params['OptPar_Opt']
#    OptParam_InputSequence = OptParam['InputSeqRange'][4]
#    OptParam_BatchSize = OptParam['BatchSizeRange'][0]
#    OptParam_NumSeqStateUnit = OptParam['InputSeqRange'][0]        
#    OptParam_SeqActSet = OptParam['ActiveSet_seq'][5]
#    OptParam_SeqModel = OptParam['SeqModelSet'][0]
#    OptParam_Optimizer = OptParam['Optimizer'][0]
    OptParamList = [OptParam_InputSequence, OptParam_NumSeqStateUnit, OptParam_SeqActSet, OptParam_RecSeqActSet, OptParam_DenseActSet, OptParam_Optimizer, OptParam_BatchSize]
    lstParamHistory.append(OptParamList)
    global Numiteration     
    print ('\n ') 
    print ('Iteration: ', Numiteration)   
    #%% Data slicing 

    tmpDataSet_Y = []
    tmpDataSet_X = []
    
    for key in DataLoad.keys():
        tmpDataCurrent = DataLoad[key]
        tmpLen = np.shape(tmpDataCurrent)[0]
        tmpEndPoint = tmpLen + OptParam_InputSequence - tmpLen%OptParam_InputSequence
        DataCurrent = np.zeros((tmpEndPoint,6))
        DataCurrent[:tmpLen,:] = tmpDataCurrent
        DataCurrent[tmpLen:,:] = tmpDataCurrent[tmpLen-1,:]
        DataSize = DataCurrent.shape[0]
        for index in range(0,DataSize-OptParam_InputSequence):        
            tmpDataSet_X.append(DataCurrent[index:index+OptParam_InputSequence,1:-1])
            tmpDataSet_Y.append(DataCurrent[index+OptParam_InputSequence,0])            
        del tmpDataCurrent    
    DataSet_X = np.array(tmpDataSet_X); del tmpDataSet_X
    DataSet_Y = np.array(tmpDataSet_Y); del tmpDataSet_Y
        
        
    [DataSet_X_Norm, Data_X_Max, Data_X_Min] = Idm_Lib.NormColArry(DataSet_X)
    [DataSet_Y_Norm, Data_Y_Max, Data_Y_Min] = Idm_Lib.NormColArry(DataSet_Y)
    
    TrainConfig_TrainRatio = 0.8
    tmp_DataLen = np.shape(DataSet_X_Norm)[0]
    tmp_lstTrainSet = list(range(tmp_DataLen))[0:int(tmp_DataLen*TrainConfig_TrainRatio)]
    tmp_lstValidSet = list(range(tmp_DataLen))[int(tmp_DataLen*TrainConfig_TrainRatio):] 
    
    #DataNorm_Den = DataSet_X_Max-DataSet_X_Min;
    
    x_train = DataSet_X_Norm[tmp_lstTrainSet,:,:];
    y_train = DataSet_Y_Norm[tmp_lstTrainSet];
    
    x_valid = DataSet_X_Norm[tmp_lstValidSet,:,:];
    y_valid = DataSet_Y_Norm[tmp_lstValidSet];

    start = time.time()
    
    #%% Model structing    
    ModelConfig_NumFeature = 4
#    K.clear_session()
    model = Sequential()
    
#    if OptParam_SeqModel == 'SimpleRnn':
#        ModSeq1 = SimpleRNN(OptParam_NumSeqStateUnit, activation=OptParam_SeqActSet, recurrent_activation = OptParam_RecSeqActSet, return_sequences=True,input_shape=(OptParam_InputSequence,ModelConfig_NumFeature))
#    else:
    ModSeq1 = LSTM(OptParam_NumSeqStateUnit, activation=OptParam_SeqActSet, recurrent_activation = OptParam_RecSeqActSet, return_sequences=True,input_shape=(OptParam_InputSequence,ModelConfig_NumFeature))
    model.add(ModSeq1)
    ModDens1 = Dense(1, activation=OptParam_DenseActSet,input_shape=(OptParam_InputSequence,OptParam_NumSeqStateUnit))
    model.add(ModDens1)
    ModReshape1 = Reshape((OptParam_InputSequence,))
    model.add(ModReshape1)
    ModDens2 = Dense(1, activation=OptParam_DenseActSet,input_shape=(1,OptParam_InputSequence))
    model.add(ModDens2)    
    model.compile(loss='mse', optimizer=OptParam_Optimizer)       
#    print(OptParam_LearnRate)
#    model.optimizer.lr = OptParam_LearnRate/100
    #%% Train and validate model
   
#    history_model = model.fit(x_train, y_train, epochs=40, batch_size=OptParam_BatchSize,validation_data=(x_valid, y_valid), verbose=0, shuffle=False)    
    history_model = model.fit(x_train, y_train,
                              batch_size=OptParam_BatchSize, epochs=15, shuffle=True,
                              validation_data=(x_valid, y_valid))
    lstModelHistory.append(history_model)
    score=history_model.history['val_loss'][-1]
    if math.isnan(score):
        score = 1
    print(' --->  MSE: ', score)        
    
    model.save('ModelData/Rnn_Braking(%d).h5' %(Numiteration))
    Numiteration = Numiteration +1    

    end = time.time()
    print('Elapsed time: ', end-start)
    return {'loss': score, 'status': STATUS_OK}
#%% Optimize
TPE_Iteration = 1000
##########################################################################################################
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=TPE_Iteration) #algo=tpe.suggest,rand.suggest
#################################lst#########################################################################
##%% Save result
#import pickle
#
##HyperOptHistory = {'History':[], 'OptParamSet', 'OptParam', }
#DicKey = 0
#for models in lstModelHistory:
#    KerasModelMeta = models
#    print(models)
#    HyperOptHistory[DicKey] = KerasModelMeta.history
#    DicKey = DicKey+1
#    
#with open('HyperOpt_History.pickle','wb') as mysavedata:
#     pickle.dump([HyperOptHistory,best,OptParam],mysavedata)     
#mysavedata.close()
##%% Save result to matlab
#with open('HyperOpt_History.pickle','rb') as myloaddata:
#    HyperOptResult = pickle.load(myloaddata)
#
#    
#cdir = os.getcwd()
#data_dir = os.chdir('../data')
#
#ParamOptDic = HyperOptResult[0]
#
#tmpMatDic = {}
##%%
#for key in ParamOptDic.keys():
#    key_name = 'Iteration_%d' % key
#    tmpDicList = ParamOptDic[key]
#    tmpDic = {'loss':np.array(tmpDicList['loss']),'val_loss':np.array(tmpDicList['val_loss'])}
#    tmpMatDic[key_name] = tmpDic
#    
#io.savemat('ParamOptResult.mat',tmpMatDic)
