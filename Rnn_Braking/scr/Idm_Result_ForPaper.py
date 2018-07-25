# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:31:46 2018

@author: Kyunghan
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import lib_sbpac
import Idm_Lib 
import os
import math
import scipy.io
import Idm_PredictionModel_ClusteringModel as ClusterResult
#%% Import data and normalization
Color = lib_sbpac.get_ColorSet()
Color['Driver1'] = Color['BP']
Color['Driver2'] = Color['RP']
Color['Driver3'] = Color['GP']
cdir = os.getcwd()
data_dir = os.chdir('../data')

DataLoad = scipy.io.loadmat('TestUpDataVariRan.mat');
del DataLoad['__globals__'];del DataLoad['__header__'];del DataLoad['__version__'];
TrainConfig_NumDataSet = DataLoad['DataLength'][0,0];
del DataLoad['DataLength'];

os.chdir(cdir)

with open('PredictionResult.pickle','rb') as myloaddata:
     PredictionResult = pickle.load(myloaddata)
     myloaddata.close()

with open('PredictionResult_Clust_clustering.pickle','rb') as myloaddata:
     PredictionResult_Clust_over = pickle.load(myloaddata)
     PredictionResult_Clust_under = pickle.load(myloaddata)
     PredictionResult_Clust = pickle.load(myloaddata)
     PredictionResult_Merge = pickle.load(myloaddata)
     myloaddata.close()
          
     
# Parameter arrangement
PlotIndex = {'Driver1':np.arange(1,101)}
PlotIndex['Driver2'] = np.arange(101,201)
PlotIndex['Driver3'] = np.arange(201,298)

ParamResult = []
Param_Cluster = {}
for key in sorted(PlotIndex.keys()):
    DriverIndex = PlotIndex[key]                    
    Param_Driver = {'MaxTp':[],'Slope':[],'MaxAcc':[],'AccDiff':[],'DataCase':[],'AccMaxPt':[],'Cluster':[],'AccDiffMaxPt':[]}
    for i in range(len(DriverIndex)):
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
        VehDataArray = DataLoad[tmpDataCase]
        ParamResult_Case = Idm_Lib.RefCalc(VehDataArray)
        ParamResult.append(ParamResult_Case)
        Param_Driver['MaxTp'].append(ParamResult_Case[0]['Par_MaxPoint']/10)
        Param_Driver['Slope'].append(ParamResult_Case[0]['Par_Slope']*10)
        Param_Driver['MaxAcc'].append(ParamResult_Case[0]['Par_MaxAcc'])
        Param_Driver['AccDiff'].append(ParamResult_Case[0]['Par_AccDiff'])
        Param_Driver['AccMaxPt'].append(ParamResult_Case[0]['Par_AccMaxPt'])
        Param_Driver['AccDiffMaxPt'].append(ParamResult_Case[0]['Par_AccMaxDiff'])
        Param_Driver['DataCase'].append(tmpDataCase)
        Param_Driver['Cluster'].append(ClusterResult.ClusterIndex[tmpCaseIndex-1])
    glb_driver = 'Param_%s' %key
    globals()[glb_driver] = Param_Driver     
#%% 1. Prediction result for driver cases - wide figure
PlotIndex = {'Driver1':[1,2,3]}
PlotIndex['Driver2'] = [146,147,148]
PlotIndex['Driver3'] = [232,233,236]

Color['Driver1'] = Color['BP']
Color['Driver2'] = Color['RP']
Color['Driver3'] = Color['GP']


VelCoast = []
for key in sorted(PlotIndex.keys()):
    print(key)    
    DriverIndex = PlotIndex[key]
    fig_name = 'fig2_wide_%s.png' % key
    print(fig_name)
    fig = plt.figure(figsize = (12,8))
    ax1 = [];ax2 = [];ax3 = []
    fig_color = Color[key]
    for i in range(len(DriverIndex)):
        ax1.append(fig.add_subplot(2,3,1 + i))                
        ax2.append(fig.add_subplot(4,3,7 + i))
        ax3.append(fig.add_subplot(4,3,10 + i))        
        ax1[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax2[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax3[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
        PredicArray = PredictionResult[tmpDataCase]        
        TimePrediction = PredicArray[:,-1]
        VehDataArray = DataLoad[tmpDataCase]
        TimeVehicle = VehDataArray[:,4] - 0.01
        tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]
        VelCoast.append(VehDataArray[0,1])

        ax1[i].plot(TimePrediction,PredicArray[:,0], label = 'AccPredic', c=fig_color[4])    
        ax1[i].plot(TimePrediction,PredicArray[:,3], ls = '-.', label = 'AccRefPredic', c=fig_color[1])
#        ax1[i].plot(TimeVehicle,tmpAccRefCalc, ls = '-.', label = 'AccRefPredic', c=fig_color[3])
        ax1[i].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
        ax1[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', ls = '--', c = Color['WP'][3])
#        ax1[i].set_ylim([-5,0])
        ax2[i].plot(TimePrediction,PredicArray[:,1]*3.6, label = 'VelPredic', c=fig_color[4])
        ax2[i].plot(TimeVehicle,VehDataArray[:,1]*3.6, label = 'Vel',ls = '--', c = Color['WP'][5])
        ax3[i].plot(TimePrediction,PredicArray[:,2], label = 'DisPredic', c=fig_color[4])
        ax3[i].plot(TimeVehicle,VehDataArray[:,2], label = 'Dis',ls = '--', c = Color['WP'][5])           
    ylim_acc = min(ax1[0].get_ylim(), ax1[1].get_ylim(), ax1[2].get_ylim())
    ylim_vel = min(ax2[0].get_ylim(), ax2[1].get_ylim(), ax2[2].get_ylim())
    ylim_dis = min(ax3[0].get_ylim(), ax3[1].get_ylim(), ax3[2].get_ylim())    
    for i in range(len(DriverIndex)):
        ax1[i].set_ylim(ylim_acc);
        ax2[i].set_ylim(ylim_vel);
        ax3[i].set_ylim(ylim_dis);        
        title_str = 'Case_%d' % (i+1)
        ax1[i].set_title(title_str)
        ax3[i].set_xlabel('Time [s]')
        if i == 0:
            ax1[i].set_ylabel('Acceleration [m/s$^2$]')
            ax2[i].set_ylabel('Velocity [km/h]')
            ax3[i].set_ylabel('Distance [m]')
            ax1[i].legend();ax2[i].legend();ax3[i].legend();
           
    plt.show()
    plt.savefig(fig_name, format='png', dpi=500)    
#%%
for key in sorted(PlotIndex.keys()):
    print(key)    
    DriverIndex = PlotIndex[key]
    fig_name = 'fig2_col_%s.png' % key
    print(fig_name)
    fig = plt.figure(figsize = (6,8))
    ax1 = [];ax2 = [];ax3 = []
    fig_color = Color[key]
    for i in range(len(DriverIndex)):
        ax1.append(fig.add_subplot(2,3,1 + i))                
        ax2.append(fig.add_subplot(4,3,7 + i))
        ax3.append(fig.add_subplot(4,3,10 + i))        
        ax1[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax2[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax3[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
        PredicArray = PredictionResult[tmpDataCase]        
        TimePrediction = PredicArray[:,-1]
        VehDataArray = DataLoad[tmpDataCase]
        TimeVehicle = VehDataArray[:,4] - 0.01
        tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]
        VelCoast.append(VehDataArray[0,1])

        ax1[i].plot(TimePrediction,PredicArray[:,0], label = 'AccPredic', c=fig_color[4])    
        ax1[i].plot(TimePrediction,PredicArray[:,3], ls = '-.', label = 'AccRefPredic', c=fig_color[1])
#        ax1[i].plot(TimeVehicle,tmpAccRefCalc, ls = '-.', label = 'AccRefPredic', c=fig_color[3])
        ax1[i].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
        ax1[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', ls = '--', c = Color['WP'][3])
#        ax1[i].set_ylim([-5,0])
        ax2[i].plot(TimePrediction,PredicArray[:,1]*3.6, label = 'VelPredic', c=fig_color[4])
        ax2[i].plot(TimeVehicle,VehDataArray[:,1]*3.6, label = 'Vel',ls = '--', c = Color['WP'][5])
        ax3[i].plot(TimePrediction,PredicArray[:,2], label = 'DisPredic', c=fig_color[4])
        ax3[i].plot(TimeVehicle,VehDataArray[:,2], label = 'Dis',ls = '--', c = Color['WP'][5])    
        
    ylim_acc = min(ax1[0].get_ylim(), ax1[1].get_ylim(), ax1[2].get_ylim());
    ylim_vel = min(ax2[0].get_ylim(), ax2[1].get_ylim(), ax2[2].get_ylim())
    ylim_dis = min(ax3[0].get_ylim(), ax3[1].get_ylim(), ax3[2].get_ylim())    
    for i in range(len(DriverIndex)):
        ax1[i].set_ylim([ylim_acc[0]-1.5, ylim_acc[1]]);
        ax2[i].set_ylim([ylim_vel[0], ylim_vel[1]+20]);
        ax3[i].set_ylim(ylim_dis);        
        title_str = 'Case_%d' % (i+1)
        ax1[i].set_title(title_str)
        ax3[i].set_xlabel('Time [s]')
        if i == 0:
            ax1[i].set_ylabel('Acceleration [m/s$^2$]')
            ax2[i].set_ylabel('Velocity [km/h]')
            ax3[i].set_ylabel('Distance [m]')
            ax1[i].legend();ax2[i].legend();ax3[i].legend();
        else:
            ax1[i].set_yticklabels([''])
            ax2[i].set_yticklabels([''])
            ax3[i].set_yticklabels([''])
    plt.show()
    plt.savefig(fig_name, format='png', dpi=500)    
#%% 2. Prediction result for driver cases - clustering
    PlotIndex = {'Driver1':[1,2,3]}
    PlotIndex['Driver2'] = [146,147,148]
    PlotIndex['Driver3'] = [232,233,236]
    
    Color['Driver1'] = Color['BP']
    Color['Driver2'] = Color['RP']
    Color['Driver3'] = Color['GP']
    
    plt.close('all')
    VelCoast = []
    for key in sorted(PlotIndex.keys()):
        print(key)    
        DriverIndex = PlotIndex[key]
        fig_name = 'fig8_%s.png' % key
        print(fig_name)
        fig = plt.figure(figsize = (8,8))
        ax1 = [];ax2 = [];ax3 = []
        fig_color = Color[key]
        for i in range(len(DriverIndex)):
            ax1.append(fig.add_subplot(2,3,1 + i))                
            ax2.append(fig.add_subplot(4,3,7 + i))
            ax3.append(fig.add_subplot(4,3,10 + i))        
            ax1[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
            ax2[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
            ax3[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
            tmpCaseIndex = DriverIndex[i]
            tmpDataCase = "CaseData%d" % tmpCaseIndex
            tmpCluIndex = Param_Cluster[tmpDataCase]
            PredicArray_c2 = PredictionResult_Clust_c2[tmpDataCase]        
            PredicArray_c1 = PredictionResult_Clust_c1[tmpDataCase]        
            PredicArray_Blended = PredictionResult_Clust[tmpDataCase]        
            PredicArray_Merge = PredictionResult_Merge[tmpDataCase]    
            PredicArray_Orig = PredictionResult[tmpDataCase]    
            TimePrediction = PredicArray_c2[:,-1]
            VehDataArray = DataLoad[tmpDataCase]
            TimeVehicle = VehDataArray[:,4] - 0.01
            tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]
            VelCoast.append(VehDataArray[0,1])
#            ax1[i].plot(TimePrediction,PredicArray_c2[:,0], ls = '--', label = 'AccPredic_c2_under', c=fig_color[4])    
#            ax1[i].plot(TimePrediction,PredicArray_c1[:,0], ls = '-.',label = 'AccPredic_c1_over', c=fig_color[4])    
#            ax1[i].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=fig_color[4])    
#            ax1[i].plot(TimePrediction,PredicArray_Merge[:,0], label = 'AccPredic_merge', c=fig_color[4])    
            ax1[i].plot(TimePrediction,PredicArray_c2[:,0], ls = '--', label = 'AccPredic_c2_under', c=Color['RP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_c1[:,0], ls = '-.',label = 'AccPredic_c1_over', c=Color['RP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=Color['BP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_Merge[:,0], label = 'AccPredic_merge', c=Color['GP'][4])    
            ax1[i].plot(TimePrediction,PredicArray_Orig[:,0], label = 'AccPredic_org', c=Color['PP'][4])    
            
            ax1[i].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
#            ax1[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', ls = '--', c = Color['WP'][3])
    
            ax2[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', c = Color['WP'][3])
            ax2[i].plot(TimePrediction,PredicArray_Blended[:,3], label = 'AccRef_blnd', c=Color['BP'][4])    
#            ax2[i].plot(TimePrediction,PredicArray_Merge[:,3], label = 'AccRef_merge', c=Color['GP'][4])    
            ax2[i].plot(TimePrediction,PredicArray_Orig[:,3], label = 'AccRef_org', c=Color['PP'][4])    
            
            ax3[i].plot(TimePrediction,PredicArray_c2[:,2], label = 'DisPredic_blnd', c=fig_color[4])
            ax3[i].plot(TimeVehicle,VehDataArray[:,2], label = 'Dis',ls = '--', c = Color['WP'][5])           
        ylim_acc = min(ax1[0].get_ylim(), ax1[1].get_ylim(), ax1[2].get_ylim())
        ylim_vel = min(ax2[0].get_ylim(), ax2[1].get_ylim(), ax2[2].get_ylim())
        ylim_dis = min(ax3[0].get_ylim(), ax3[1].get_ylim(), ax3[2].get_ylim())    
        for i in range(len(DriverIndex)):
            tmpCaseIndex = DriverIndex[i]
            tmpDataCase = "CaseData%d" % tmpCaseIndex
            tmpCluIndex = Param_Cluster[tmpDataCase]
            ax1[i].set_ylim(ylim_acc);
            ax2[i].set_ylim(ylim_vel);
            ax3[i].set_ylim(ylim_dis);        
            title_str = 'Case_%d' % (i+1)
            ax1[i].set_title(title_str)
            ax3[i].set_xlabel('Time [s]')
            if i == 0:
                ax1[i].set_ylabel('Acceleration [m/s$^2$]')
                ax2[i].set_ylabel('Velocity [km/h]')
                ax3[i].set_ylabel('Distance [m]')
                ax1[i].legend();ax2[i].legend();ax3[i].legend();
        plt.show()  
#%%
for key in sorted(PlotIndex.keys()):
    print(key)    
    DriverIndex = PlotIndex[key]
    fig_name = 'fig8_col_%s.png' % key
    print(fig_name)
    fig = plt.figure(figsize = (6,8))
    ax1 = [];ax2 = [];ax3 = []
    fig_color = Color[key]
    for i in range(len(DriverIndex)):
        ax1.append(fig.add_subplot(2,3,1 + i))                
        ax2.append(fig.add_subplot(4,3,7 + i))
        ax3.append(fig.add_subplot(4,3,10 + i))        
        ax1[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax2[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax3[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
        PredicArray_Under = PredictionResult_Clust_under[tmpDataCase]        
        PredicArray_Over = PredictionResult_Clust_over[tmpDataCase]        
        PredicArray_Blended = PredictionResult_Clust[tmpDataCase]        
        TimePrediction = PredicArray_Under[:,-1]
        VehDataArray = DataLoad[tmpDataCase]
        TimeVehicle = VehDataArray[:,4] - 0.01
        tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]
        ax1[i].plot(TimePrediction,PredicArray_Under[:,0], ls = '--', label = 'AccPredic_under', c=fig_color[3])    
        ax1[i].plot(TimePrediction,PredicArray_Over[:,0], ls = '-.',label = 'AccPredic_over', c=fig_color[5])    
        ax1[i].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=fig_color[4])    
        ax1[i].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
        ax1[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', ls = '--', c = Color['WP'][3])
    
        ax2[i].plot(TimePrediction,PredicArray_Under[:,1]*3.6, label = 'VelPredic_blnd', c=fig_color[4])
        ax2[i].plot(TimeVehicle,VehDataArray[:,1]*3.6, label = 'Vel',ls = '--', c = Color['WP'][5])
        
        ax3[i].plot(TimePrediction,PredicArray_Under[:,2], label = 'DisPredic_blnd', c=fig_color[4])
        ax3[i].plot(TimeVehicle,VehDataArray[:,2], label = 'Dis',ls = '--', c = Color['WP'][5])  
        if i == 0:
            ax1[i].set_ylabel('Acceleration [m/s$^2$]')
            ax2[i].set_ylabel('Velocity [km/h]')
            ax3[i].set_ylabel('Distance [m]')
            ax1[i].legend()
            ax2[i].legend(loc = 2)
            ax3[i].legend(loc = 2)            
        else:
            ax1[i].set_yticklabels([''])
            ax2[i].set_yticklabels([''])
            ax3[i].set_yticklabels([''])
            
    ylim_acc = min(ax1[0].get_ylim(), ax1[1].get_ylim(), ax1[2].get_ylim());
    ylim_vel = min(ax2[0].get_ylim(), ax2[1].get_ylim(), ax2[2].get_ylim())
    ylim_dis = min(ax3[0].get_ylim(), ax3[1].get_ylim(), ax3[2].get_ylim())    
    for i in range(len(DriverIndex)):
        ax1[i].set_ylim([ylim_acc[0]-1.5, ylim_acc[1]]);
        ax2[i].set_ylim([ylim_vel[0], ylim_vel[1]+20]);
        ax3[i].set_ylim(ylim_dis);        
        title_str = 'Case_%d' % (i+1)
        ax1[i].set_title(title_str)
        ax3[i].set_xlabel('Time [s]')

    plt.show()
    plt.savefig(fig_name, format='png', dpi=500)    
#%% 2. Driving characteristics parameter - python + PPT
PlotIndex = {'Driver1':[1,2,3]}
PlotIndex['Driver2'] = [146,147,148]
PlotIndex['Driver3'] = [232,233,236]

ParamResult = []
lines = []
ls_set = ['-','--','-.']

fig = plt.figure(figsize = (8,6)); ax1 = []; ax2 = [];
fig_name = 'fig3_wide.png'
fig_index = 0;
for key in sorted(PlotIndex.keys()):
    DriverIndex = PlotIndex[key]                    
    ax1.append(fig.add_subplot(2,3,1 + fig_index))            
    ax2.append(fig.add_subplot(2,3,4 + fig_index))    
    ax1[fig_index].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
    ax2[fig_index].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
    fig_color = Color[key]
    for i in range(len(DriverIndex)):
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
#        PredicArray = PredictionResult[tmpDataCase]
#        TimePrediction = PredicArray[:,-1]
        VehDataArray = DataLoad[tmpDataCase]
        ParamResult_Case = Idm_Lib.RefCalc(VehDataArray)
        TimeVehicle = VehDataArray[:,4] - 0.01

        ParamResult.append(ParamResult_Case)
        
        VehData_VelDiff = ParamResult_Case[2]
        Param_MaxPoint = ParamResult_Case[0]['Par_MaxPoint']
        Param_BrkPoint = 0
        Param_Slope = ParamResult_Case[0]['Par_Slope']
        tmpX = np.arange(Param_BrkPoint,Param_MaxPoint+1)
        tmpY = Param_Slope*tmpX + VehDataArray[0,0]
        
        ax1[fig_index].plot(TimeVehicle, VehDataArray[:,0], ls = ls_set[i], label = 'Acc', c = Color['WP'][5+i])
        ax1[fig_index].plot(TimeVehicle[tmpX], tmpY, ls = ls_set[i], label = 'Slope', c = fig_color[1+i])        
        ax1[fig_index].plot(TimeVehicle[Param_MaxPoint], VehDataArray[Param_MaxPoint,0], 'o', c = fig_color[1+i])        
        line, = ax2[fig_index].plot(TimeVehicle, VehData_VelDiff, ls = ls_set[i], label = 'VelDiff', c = Color['WP'][5+i])
        lines.append(line)
        ax2[fig_index].plot(TimeVehicle[Param_MaxPoint], VehData_VelDiff[Param_MaxPoint], 'o', c = fig_color[1+i])
    
    ax2[fig_index].set_xlabel('Time [s]')            
    ax1[fig_index].set_title(key)        
    fig_index = fig_index + 1

ax1[0].legend(['acc',r'$\alpha$','acc$_{point}$'])
ax2[0].legend((lines[0],lines[1],lines[2]),['Case 1','Case 2','Case 3'])
ax1[0].set_ylabel('Acceleration [m/s$^2$]')
ax2[0].set_ylabel('$\Delta$Velocity  [m/s]')

plt.savefig(fig_name, format='png', dpi=600)     
#%%
fig = plt.figure(figsize = (6,5)); ax1 = []; ax2 = [];
fig_name = 'fig3_col_decelchar.png'
fig_index = 0;
for key in sorted(PlotIndex.keys()):
    DriverIndex = PlotIndex[key]                    
    ax1.append(fig.add_subplot(2,3,1 + fig_index))            
    ax2.append(fig.add_subplot(2,3,4 + fig_index))    
    ax1[fig_index].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
    ax2[fig_index].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
    fig_color = Color[key]
    for i in range(len(DriverIndex)):
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
#        PredicArray = PredictionResult[tmpDataCase]
#        TimePrediction = PredicArray[:,-1]
        VehDataArray = DataLoad[tmpDataCase]
        ParamResult_Case = Idm_Lib.RefCalc(VehDataArray)
        TimeVehicle = VehDataArray[:,4] - 0.01

        ParamResult.append(ParamResult_Case)
        
        VehData_VelDiff = ParamResult_Case[2]
        Param_MaxPoint = ParamResult_Case[0]['Par_MaxPoint']
        Param_BrkPoint = 0
        Param_Slope = ParamResult_Case[0]['Par_Slope']
        tmpX = np.arange(Param_BrkPoint,Param_MaxPoint+1)
        tmpY = Param_Slope*tmpX + VehDataArray[0,0]
        
        ax1[fig_index].plot(TimeVehicle, VehDataArray[:,0], ls = ls_set[i], label = 'Acc', c = Color['WP'][5+i])
        ax1[fig_index].plot(TimeVehicle[tmpX], tmpY, ls = ls_set[i], label = 'Slope', c = fig_color[1+i])        
        ax1[fig_index].plot(TimeVehicle[Param_MaxPoint], VehDataArray[Param_MaxPoint,0], 'o', c = fig_color[1+i])        
        line, = ax2[fig_index].plot(TimeVehicle, VehData_VelDiff, ls = ls_set[i], label = 'VelDiff', c = Color['WP'][5+i])
        lines.append(line)
        ax2[fig_index].plot(TimeVehicle[Param_MaxPoint], VehData_VelDiff[Param_MaxPoint], 'o', c = fig_color[1+i])
    
    ax2[fig_index].set_xlabel('Time [s]')            
    ax1[fig_index].set_title(key)        
    fig_index = fig_index + 1

y_lim_vel = ax2[0].get_ylim()
ax2[0].set_ylim([y_lim_vel[0], y_lim_vel[1] + 1.5])
ax2[1].set_ylim([y_lim_vel[0], y_lim_vel[1] + 1.5])
ax2[2].set_ylim([y_lim_vel[0], y_lim_vel[1] + 1.5])

ax1[0].legend(['acc',r'$\alpha$','acc$_{point}$'])
ax2[0].legend((lines[0],lines[1],lines[2]),['Case 1','Case 2','Case 3'])
ax1[0].set_ylabel('Acceleration [m/s$^2$]')
ax2[0].set_ylabel('$\Delta$Velocity  [m/s]')

plt.savefig(fig_name, format='png', dpi=600)
#%% 3. Driving characteristics analysis for driver
fig = plt.figure(figsize = (4,3))
fig_name = 'fig4_maxacc.png'
ax1 = fig.add_axes([0.3,0.2,0.65,0.75])
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.scatter(Param_Driver1['AccDiff'],Param_Driver1['MaxAcc'],c=Color['Driver1'][5],alpha = 0.3,marker = 'o')
ax1.scatter(Param_Driver2['AccDiff'],Param_Driver2['MaxAcc'],c=Color['Driver2'][8],alpha = 0.3,marker = '^')
ax1.scatter(Param_Driver3['AccDiff'],Param_Driver3['MaxAcc'],c=Color['Driver3'][3],alpha = 0.3,marker = 's')
ax1.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel('Maximum acceleration [m/s$^2$]')
plt.savefig(fig_name, format='png', dpi=600)        

fig = plt.figure(figsize = (4,3))
ax1 = fig.add_axes([0.3,0.2,0.65,0.75])
fig_name = 'fig4_maxpoint.png'
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.scatter(Param_Driver1['AccDiff'],Param_Driver1['MaxTp'],c=Color['Driver1'][5],alpha = 0.3,marker = 'o')
ax1.scatter(Param_Driver2['AccDiff'],Param_Driver2['MaxTp'],c=Color['Driver2'][8],alpha = 0.3,marker = '^')
ax1.scatter(Param_Driver3['AccDiff'],Param_Driver3['MaxTp'],c=Color['Driver3'][3],alpha = 0.3,marker = 's')
ax1.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel('Deceleration time $t_{point}$ [s]')
plt.savefig(fig_name, format='png', dpi=600)        

fig = plt.figure(figsize = (4,3))
fig_name = 'fig4_decelslope.png'
ax1 = fig.add_axes([0.3,0.2,0.65,0.75])
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.scatter(Param_Driver1['AccDiff'],Param_Driver1['Slope'],c=Color['Driver1'][5],alpha = 0.3,marker = 'o')
ax1.scatter(Param_Driver2['AccDiff'],Param_Driver2['Slope'],c=Color['Driver2'][8],alpha = 0.3,marker = '^')
ax1.scatter(Param_Driver3['AccDiff'],Param_Driver3['Slope'],c=Color['Driver3'][3],alpha = 0.3,marker = 's')
ax1.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel(r'Deceleration slope $\alpha$ [m/s$^3$]')
plt.savefig(fig_name, format='png', dpi=600)
#%% 3. Driver characteristic analysis - wide version
fig = plt.figure(figsize = (12,3.5))
fig_name = 'fig4_wide_parameter.png'

ax1 = fig.add_subplot(1,3,1)
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.scatter(Param_Driver1['AccDiff'],Param_Driver1['MaxAcc'],c=Color['Driver1'][5],alpha = 0.3,marker = 'o')
ax1.scatter(Param_Driver2['AccDiff'],Param_Driver2['MaxAcc'],c=Color['Driver2'][8],alpha = 0.3,marker = '^')
ax1.scatter(Param_Driver3['AccDiff'],Param_Driver3['MaxAcc'],c=Color['Driver3'][3],alpha = 0.3,marker = 's')
ax1.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel('Maximum acceleration [m/s$^2$]')
ax1.set_title('$(a)$')

ax2 = fig.add_subplot(1,3,2)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.scatter(Param_Driver1['AccDiff'],Param_Driver1['AccMaxPt'],c=Color['Driver1'][5],alpha = 0.3,marker = 'o')
ax2.scatter(Param_Driver2['AccDiff'],Param_Driver2['AccMaxPt'],c=Color['Driver2'][8],alpha = 0.3,marker = '^')
ax2.scatter(Param_Driver3['AccDiff'],Param_Driver3['AccMaxPt'],c=Color['Driver3'][3],alpha = 0.3,marker = 's')
ax2.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax2.set_xlabel('Acc index [m/s$^2$]')
ax2.set_ylabel('Acceleration at $t_{point}$ [m/s$^2$]')
ax2.set_title('$(b)$')

ax3 = fig.add_subplot(1,3,3)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.scatter(Param_Driver1['AccDiff'],Param_Driver1['Slope'],c=Color['Driver1'][5],alpha = 0.3,marker = 'o')
ax3.scatter(Param_Driver2['AccDiff'],Param_Driver2['Slope'],c=Color['Driver2'][8],alpha = 0.3,marker = '^')
ax3.scatter(Param_Driver3['AccDiff'],Param_Driver3['Slope'],c=Color['Driver3'][3],alpha = 0.3,marker = 's')
ax3.legend(['Driver 1', 'Driver 2', 'Driver 3'])
ax3.set_xlabel('Acc index [m/s$^2$]')
ax3.set_ylabel(r'Deceleration slope $\alpha$ [m/s$^3$]')
ax3.set_title('$(c)$')

plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.savefig(fig_name, format='png', dpi=600)
#%% 4. Braking profile for Deceleration characteristics and prediction results
min_acc_diff = 1.5
max_acc_diff = 1.6
x_index = 'AccDiff'
y_index = 'AccDiffMaxPt'
        
tmpAccDiff = np.array(Param_Driver3[x_index])
index_def = np.where(np.all([(tmpAccDiff > min_acc_diff),(tmpAccDiff < max_acc_diff)], axis = 0))[0]
tmpAccMaxPnt = np.array(Param_Driver3[y_index])[index_def]
tmpIndex_over = index_def[np.argmax(tmpAccMaxPnt)]
tmpDataCaseOver = Param_Driver3['DataCase'][tmpIndex_over]

tmpAccDiff = np.array(Param_Driver3[x_index])
tmpIndex_under = np.where(np.all([(tmpAccDiff > min_acc_diff),(tmpAccDiff < max_acc_diff)], axis = 0))[0]
tmpAccMaxPnt = np.array(Param_Driver3[y_index])[tmpIndex_under]
tmptmpIndex_under = tmpIndex_under[np.argmin(tmpAccMaxPnt)]
tmpDataCaseUnder = Param_Driver3['DataCase'][tmptmpIndex_under]

over_x_array = np.array([])
over_y_array = np.array([])
under_x_array = np.array([])
under_y_array = np.array([])
for i in range(3):
    tmpParamDriver = globals()['Param_Driver%d' %(i+1)]
    tmpCluster = np.array(tmpParamDriver['Cluster'])
    tmpClusterIndex_over = tmpCluster == 1
    tmpClusterIndex_under = tmpCluster == 2
    tmpYval_over = np.array(tmpParamDriver[y_index])[tmpClusterIndex_over]
    tmpXval_over = np.array(tmpParamDriver[x_index])[tmpClusterIndex_over]
    tmpYval_under = np.array(tmpParamDriver[y_index])[tmpClusterIndex_under]
    tmpXval_under = np.array(tmpParamDriver[x_index])[tmpClusterIndex_under]
    globals()['Yval_over_%d' % (i+1)] = tmpYval_over
    globals()['Xval_over_%d' % (i+1)] = tmpXval_over
    globals()['Yval_under_%d' % (i+1)] = tmpYval_under
    globals()['Xval_under_%d' % (i+1)] = tmpXval_under
    over_x_array = np.concatenate((over_x_array, tmpXval_over))
    over_y_array = np.concatenate((over_y_array, tmpYval_over))
    under_x_array = np.concatenate((under_x_array, tmpXval_under))
    under_y_array = np.concatenate((under_y_array, tmpYval_under))

fig = plt.figure(figsize = (4,3.5))
fig_name = 'fig4_AccMaxPoint_CriticalLine.png'
ax1 = fig.add_axes([0.2,0.2,0.65,0.75])
ax1.scatter(Param_Driver1[x_index],Param_Driver1[y_index],c=Color['Driver1'][5],alpha = 0.2,marker = 'o')
ax1.scatter(Param_Driver2[x_index],Param_Driver2[y_index],c=Color['Driver2'][8],alpha = 0.2,marker = '^')
ax1.scatter(Param_Driver3[x_index],Param_Driver3[y_index],c=Color['Driver3'][3],alpha = 0.2,marker = 's')
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax1.plot(ClusterResult.x_data,np.zeros(297),label = 'Critical Line')
ax1.scatter(Param_Driver1[x_index][tmpIndex_over],Param_Driver3[y_index][tmpIndex_over],c='black',alpha = 1,label = 'Over Case')
ax1.scatter(Param_Driver1[x_index][tmptmpIndex_under],Param_Driver3[y_index][tmptmpIndex_under],c='black',alpha = 1, label = 'Under Case',marker = '^')
ax1.legend(loc = 'upper left')
ax1.set_xlabel('Acc index [m/s$^2$]')
ax1.set_ylabel(r'${\Delta}$Acc max point [m/s$^2$]')
plt.savefig(fig_name, format='png', dpi=600)
#%%
key_index = [tmpDataCaseOver, tmpDataCaseUnder]

fig = plt.figure(figsize = (6,4))
fig_name = 'fig9_decel_case.png'
ax1 = [];ax2 = [];
fig_color = Color['RP']
fig_index = 0
for i in key_index:  
    print(i)
    ax1.append(fig.add_subplot(1,2,1 + fig_index))                        
    ax1[fig_index].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
    tmpDataCase = i    
    PredicArray = PredictionResult[tmpDataCase]   
    PredicArray_Blended = PredictionResult_Clust[tmpDataCase]
    PredicArray_Under = PredictionResult_Clust_under[tmpDataCase]        
    PredicArray_Over = PredictionResult_Clust_over[tmpDataCase]            
    TimePrediction = PredicArray[:,-1]
    VehDataArray = DataLoad[tmpDataCase]
    TimeVehicle = VehDataArray[:,4] - 0.01
    tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]    
    ax1[fig_index].plot(TimePrediction,PredicArray[:,0], label = 'AccPredic', c=fig_color[5])    
    ax1[fig_index].plot(TimePrediction, PredicArray[:,3], label = 'AccPredicRef', ls = '--', c = Color['WP'][3])
    ax1[fig_index].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
    ax1[fig_index].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', c = Color['WP'][1])
    ax1[fig_index].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=fig_color[4])
    ax1[fig_index].plot(TimePrediction,PredicArray_Under[:,0], ls = '--', label = 'AccPredic_under', c=fig_color[7])    
    ax1[fig_index].plot(TimePrediction,PredicArray_Over[:,0], ls = '-.',label = 'AccPredic_over', c=fig_color[6]) 
#    ax1[fig_index].plot(TimePrediction,PredicArray_Blended[:,3], label = 'AccPredicRef_blnd', c=fig_color[4])
    fig_index = fig_index + 1
ax1[0].legend()

plt.savefig(fig_name, format='png', dpi=600)   

#%%
key_index = [tmpDataCaseOver, tmpDataCaseUnder]

fig = plt.figure(figsize = (6,4))
fig_name = 'fig9_decel_case.png'
ax1 = [];ax2 = [];
fig_color = Color['RP']
fig_index = 0
for i in key_index:  
    print(i)
    ax1.append(fig.add_subplot(1,2,1 + fig_index))                        
    ax1[fig_index].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
    tmpDataCase = i
    PredicArray_Under = PredictionResult_Clust_under[tmpDataCase]        
    PredicArray_Over = PredictionResult_Clust_over[tmpDataCase]        
    PredicArray_Blended = PredictionResult_Clust[tmpDataCase]        
    PredicArray = PredictionResult[tmpDataCase]    
    TimePrediction = PredicArray_Under[:,-1]
    VehDataArray = DataLoad[tmpDataCase]
    TimeVehicle = VehDataArray[:,4] - 0.01
    tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]
    ax1[fig_index].plot(TimePrediction,PredicArray_Under[:,0], ls = '--', label = 'AccPredic_under', c=fig_color[4])    
    ax1[fig_index].plot(TimePrediction,PredicArray_Over[:,0], ls = '-.',label = 'AccPredic_over', c=fig_color[4])    
    ax1[fig_index].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=fig_color[4])    
    ax1[fig_index].plot(TimePrediction,PredicArray[:,0], label = 'AccPredic', c=fig_color[5])    

    ax1[fig_index].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
    ax1[fig_index].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', ls = '--', c = Color['WP'][3])
    
    fig_index = fig_index + 1
plt.savefig(fig_name, format='png', dpi=600)  

#%%
PlotIndex = {'Driver1':[1,2,3]}
PlotIndex['Driver2'] = [146,147,148]
PlotIndex['Driver3'] = [232,233,236]
plt.close('all')
for key in sorted(PlotIndex.keys()):
    print(key)    
    DriverIndex = PlotIndex[key]
    fig_name = 'fig8_wide_%s.png' % key
    print(fig_name)
    fig = plt.figure(figsize = (8,8))
    ax1 = [];ax2 = [];ax3 = []
    fig_color = Color[key]
    tmpBlendFac = BlendedFac[key]
    for i in range(len(DriverIndex)):
        ax1.append(fig.add_subplot(2,3,1 + i))                
        ax2.append(fig.add_subplot(4,3,7 + i))
        ax3.append(fig.add_subplot(4,3,10 + i))        
        ax1[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax2[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        ax3[i].grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
        PredicArray_Under = PredictionResult_Clust_under[tmpDataCase]        
        PredicArray_Over = PredictionResult_Clust_over[tmpDataCase]        
        PredicArray_Blended = PredictionResult_Clust[tmpDataCase]        
        PredicArray_Blended_Calc = PredicArray_Over[:,0]*tmpBlendFac + PredicArray_Under[:,0]*(1-tmpBlendFac)
        PredicArray = PredictionResult[tmpDataCase]   
        TimePrediction = PredicArray_Under[:,-1]
        VehDataArray = DataLoad[tmpDataCase]
        TimeVehicle = VehDataArray[:,4] - 0.01
        tmpAccRefCalc = -0.5*VehDataArray[:,1]*VehDataArray[:,1]/VehDataArray[:,2]
        ax1[i].plot(TimePrediction,PredicArray_Under[:,0], ls = '--', label = 'AccPredic_agg', c=fig_color[4])    
        ax1[i].plot(TimePrediction,PredicArray_Over[:,0], ls = '-.',label = 'AccPredic_dep', c=fig_color[4])    
        ax1[i].plot(TimePrediction,PredicArray_Blended[:,0], label = 'AccPredic_blnd', c=fig_color[9])
        ax1[i].plot(TimePrediction, PredicArray_Blended_Calc, ls = '--', label = 'AccPredicRef_blnd', c = fig_color[5])
        
        ax1[i].plot(TimePrediction,PredicArray[:,0], label = 'AccPredic', c=fig_color[8])
#        ax1[i].plot(TimePrediction, PredicArray[:,3], ls = '--', label = 'AccPredicRef', c = fig_color[8])
        
        ax1[i].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
#        ax1[i].plot(TimeVehicle, VehDataArray[:,0], label = 'Acc', c = Color['WP'][5])
        ax1[i].plot(TimeVehicle, VehDataArray[:,3], label = 'AccRef', ls = '--', c = Color['WP'][3])
    
        ax2[i].plot(TimePrediction,PredicArray_Blended[:,1]*3.6, label = 'VelPredic_blnd', c=fig_color[3])
        ax2[i].plot(TimePrediction,PredicArray[:,1]*3.6, label = 'VelPredic', c=fig_color[8])
        ax2[i].plot(TimeVehicle,VehDataArray[:,1]*3.6, label = 'Vel',ls = '--', c = Color['WP'][5])        
        
        ax3[i].plot(TimePrediction,PredicArray_Blended[:,2], label = 'DisPredic_blnd', c=fig_color[3])
        ax3[i].plot(TimePrediction,PredicArray[:,2], label = 'DisPredic', c=fig_color[8])
        ax3[i].plot(TimeVehicle,VehDataArray[:,2], label = 'Dis',ls = '--', c = Color['WP'][5])   
        
    ylim_acc = min(ax1[0].get_ylim(), ax1[1].get_ylim(), ax1[2].get_ylim())
    ylim_vel = min(ax2[0].get_ylim(), ax2[1].get_ylim(), ax2[2].get_ylim())
    ylim_dis = min(ax3[0].get_ylim(), ax3[1].get_ylim(), ax3[2].get_ylim())    
    for i in range(len(DriverIndex)):
        tmpCaseIndex = DriverIndex[i]
        tmpDataCase = "CaseData%d" % tmpCaseIndex
        ax1[i].set_ylim(ylim_acc);
        ax2[i].set_ylim(ylim_vel);
        ax3[i].set_ylim(ylim_dis);        
        title_str = 'Case_%d' % (i+1)
        ax1[i].set_title(title_str)
        ax3[i].set_xlabel('Time [s]')
        if i == 0:
            ax1[i].set_ylabel('Acceleration [m/s$^2$]')
            ax2[i].set_ylabel('Velocity [km/h]')
            ax3[i].set_ylabel('Distance [m]')
            ax1[i].legend();ax2[i].legend();ax3[i].legend();
    plt.show()
    plt.savefig(fig_name, format='png', dpi=500)   