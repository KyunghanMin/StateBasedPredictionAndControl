% IdmDev_LearnSILsValidation
% Date: 2018/03/12
% Description: Simulation for case iteratively (Simulink model)
clear all;close all;clc
load('Data_GB.mat')
load('ModelConfig.mat')
load('IdmParam.mat')
clearvars DataCase_*
%% MC learning and TD implementation
load('IdmParam_Integ.mat')
clearvars tmp*
% Configure parameter for one case simulation
OnlineConfig.CoastInitBuff = 0;
BrakeCri = 0.05;
Index_ModLength = 2000;
LrnFacNum = 1;
% Learning switch 
%   --- 0: Monte Carlo mean value learning
%   --- 1: Monte Carlo value update (learning rate)
LrnConfig.LrnSwt = 1;
%% Sim learning case validation
tic
% Set the simulation case
SimTestCase = 1;
%   --- Iterative test set
Index_CaseSet = SimTestCase*ones(10,1);
Indexcase = 1;
for Indexcase = 1:length(Index_CaseSet)   
    %% Parameter selection
    Index_case = Index_CaseSet(Indexcase);                
    eval(['tmpData = DataCoast.Case' num2str(Index_case) ';']);
    % Data arrangement - Vehicle
    %   --- Data_Cur = Acceleration, Velocity, Distance
    Data_Cur = [tmpData(:,1) tmpData(:,4) tmpData(:,14)];
    Data_Cur(:,3) = Data_Cur(:,3) + ModelConfig.StopDis;    
    ParamCase = Fcn_ParamSelect(Param,Index_case);       
    VehData_acc = Data_Cur(:,1);    
    VehData_vel = Data_Cur(:,2);        
    VehData_dis = Data_Cur(:,3);
    VehData_brk = tmpData(:,9);    
    VehData_aps = tmpData(:,8);
    VehData_time = (OnlineConfig.Ts:OnlineConfig.Ts:length(VehData_acc)*OnlineConfig.Ts)';    
    %   --- Enable switch - Set always 'on'
    VehData_Enable(1:Index_ModLength,1) = 1;    
    tmpTpStop = find((VehData_vel<LrnConfig.VelTerm) & (VehData_time>LrnConfig.CoastTime),1);
    if isempty(tmpTpStop)
        [tmpDummy tmpTpStop] = min(VehData_vel);
    end
    %   --- Data arrange for index buff
    VehData_acc(tmpTpStop:Index_ModLength) = VehData_acc(tmpTpStop);
    VehData_vel(tmpTpStop:Index_ModLength) = VehData_vel(tmpTpStop);
    VehData_dis(tmpTpStop:Index_ModLength) = VehData_dis(tmpTpStop);
    VehData_brk(tmpTpStop:Index_ModLength) = VehData_brk(tmpTpStop);
    VehData_aps(tmpTpStop:Index_ModLength) = VehData_aps(tmpTpStop);
    VehData_time = (OnlineConfig.Ts:OnlineConfig.Ts:Index_ModLength*OnlineConfig.Ts)';    
    SimTime = max(VehData_time);    
    VehData_CoastVel = VehData_vel(1);   
    
    % ========================== Simulation ============================= %
    sim('IdmEstimation_Cogen.slx');
    % =================================================================== %
    
    % Learning parameter update    
    BaseMap_MaxAcc_Data = LrnResult_EcpParamVec_MaxAcc.signals.values;
    BaseMap_AprStall_Data = LrnResult_EcpParamVec_AcrStall.signals.values;
    BaseMap_TpShift_Data = LrnResult_EcpParamVec_TpInit.signals.values;
    BaseMap_TpDelta_Data = LrnResult_EcpParamVec_TpDelta.signals.values;
    BaseMap_AccDiff_Data = LrnResult_EcpParamVec_AcdStall.signals.values;
    BaseMap_AdjGain_Data = LrnResult_EcpParamVec_AdjGain.signals.values;
    
    % Result storage
    LrnResult.EcpParam_MaxAcc(Indexcase) = LogData_Stop_SlcParam.signals.values(5);
    LrnResult.EcpParam_AprStall(Indexcase) = LogData_Stop_SlcParam.signals.values(4);
    LrnResult.EcpParam_TpShift(Indexcase) = LogData_Stop_SlcParam.signals.values(1);
    LrnResult.EcpParam_TpDelta(Indexcase) = LogData_Stop_SlcParam.signals.values(2);
    LrnResult.EcpParam_AcdStall(Indexcase) = LogData_Stop_SlcParam.signals.values(7);
    LrnResult.EcpParam_AdjGain(Indexcase) = LogData_Stop_SlcParam.signals.values(6);
    LrnResult.EcpParam_TpTerm(Indexcase) = LogData_Stop_LrnParam.signals.values(6);    
    LrnResult.EstProfAcc(:,Indexcase) = LogData_While_PreProf.signals.values(:,1);
    eval(['LrnResult.EstProfAcc' num2str(Indexcase) ' = LogData_While_PreProf.signals.values;'])
    LrnFacNum = LrnNum.signals.values(end);
end
%% Result plot
figure('name','','numbertitle','off')
set(gcf,'Color',[1,1,1],'position',[500 200 450 370])
plot(LrnResult.EstProfAcc,'DisplayName','LrnResult.EstProfAcc');hold on; grid on;
plot(VehData_acc)

