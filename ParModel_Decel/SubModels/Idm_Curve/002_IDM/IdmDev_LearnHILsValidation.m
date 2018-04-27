% IdmDev_LearnHILsValidation
% Date: 18/02/23
% Description: Validation using continuous driving date
clear all;close all;clc
driverset = {'KH','YK','GB'};
% Set the learning algorithm parameters
%   --- swt_lrn: Base map selection 
%       - 'on=1': Using the fitted base map
%       - 'off=1': Using the mean value base map
%   --- method: Monte Carlo learning method selection
%       - 'on=1': Mean value update
%       - 'off=1': Value update with learning rate
%   --- Alpha: Learning rate
swt_lrn_set = [1 1 0 0];
method_set = [1 0 1 0];
Alpha_set = [0.1 0.6 0.3 0.5];

% Driver 
for Index_Drivercase=1:3
    LoadFileName = ['Data_' char(driverset(Index_Drivercase)) '.mat'];
    IndexDataRange = 0;
    
    % Simulation case
    for Index_Simcase=1:length(swt_lrn_set)
        load(LoadFileName)
        load('ModelConfig.mat')
        load('IdmParam.mat')
        load('IdmParam_Integ.mat')
        % Set the learning parameter
        BaseMap_AdjGain_Data = BaseMap_AdjGain_Data + 0.01;
        LrnConfig.ExpUpdateGain = Alpha_set(Index_Simcase);
        clearvars tmp*
        % Coasting buffer for continuous simulation
        OnlineConfig.CoastInitBuff = 40;
        BrakeCri = 0.05;
        Index_ModLength = 2000;
        % Filter configuration for hils simulation
        HilFiltNum1 = 15;
        HilFiltNum2 = 9;
        LrnConfig.LrnSwt = method_set(Index_Simcase);   % 0 = mean value update, 1 = rate update
        
        % Init the learning number for mean value update
        LrnFacNum = 1;
        SimSwt_Lrn = swt_lrn_set(Index_Simcase);

        SaveFileName = ['Hils_' char(driverset(Index_Drivercase)) '_' num2str(Index_Simcase) '.mat'];
        %% Sim learning case validation
        % Learning result validation
        if SimSwt_Lrn == 0
            BaseMap_MaxAcc_Data(1:8,1) = mean(BaseMap_MaxAcc_Data);
            BaseMap_TpDelta_Data(1:8,1) = mean(BaseMap_TpDelta_Data);
            BaseMap_TpShift_Data(1:8,1) = mean(BaseMap_TpShift_Data);    
            BaseMap_AccDiff_Data(1:8,1) = mean(BaseMap_AccDiff_Data);
            BaseMap_AprStall_Data(1:8,1) = mean(BaseMap_AprStall_Data);
        end
        %% Data arrangement
        clearvars VehData_*        
        CaseSet = {'A','B','C','D'};
        CaseIndex = 1;

        VehData_acc = 0;
        VehData_velFilt = 0;
        VehData_vel = 0;
        VehData_dis = 0;
        VehData_brk = 0;
        VehData_aps = 0;
        VehData_Enable = 0;
        IndexDataRange = 0;
        % Data merging
        for CaseIndex = 1:2
        %%
            eval(['DataCase = DataCase_' char(CaseSet(CaseIndex)) ';'])

            IndexCoastNum = length(DataCase.CoastSt);
            IndexDataRange = IndexDataRange + IndexCoastNum;
            tmpVehData_acc = sgolayfilt(DataCase.Data_VehAccLong,1,HilFiltNum1);    
            tmpVehData_velFilt = sgolayfilt(DataCase.VehSpeed,1,HilFiltNum1);        
            tmpVehData_vel = DataCase.VehSpeed;
            tmpVehData_dis = DataCase.Data_Dis;
            tmpVehData_brk = sgolayfilt(DataCase.Data_DrvBrk,1,HilFiltNum2);
            tmpVehData_aps = DataCase.Data_DrvAps;
            tmpVehData_Enable = DataCase.Data_Enable;        
            VehData_acc = [VehData_acc ; tmpVehData_acc];
            VehData_velFilt = [VehData_velFilt ; tmpVehData_velFilt];
            VehData_vel = [VehData_vel ; tmpVehData_vel];
            VehData_dis = [VehData_dis ; tmpVehData_dis];
            VehData_brk = [VehData_brk ; tmpVehData_brk];
            % VehData_brk = DataCase.Data_DrvBrk;
            VehData_aps = [VehData_aps ; tmpVehData_aps];
            VehData_Enable = [VehData_Enable ; tmpVehData_Enable];    
        end
        VehData_time = (OnlineConfig.Ts:OnlineConfig.Ts:length(VehData_acc)*OnlineConfig.Ts)';    
        %% Simulation
        SimTime = max(VehData_time);    
        % ========================== Simulation ============================= %
        sim('IdmEstimation_Cogen.slx');
        % =================================================================== %

        % Learning parameter update
        HilsResult_EstProf = LogData_While_PreProf.signals.values;
        HilsResult_BrkProf = LogData_Stop_BrakingDataArray.signals.values;
        HilsResult_SlcParam = LogData_Stop_SlcParam.signals.values;    
        tmpHIlsResult_LrnParam(:,:) = LogData_Stop_LrnParam.signals.values(1,:,:);
        HilsResult_LrnParam = tmpHIlsResult_LrnParam';
        HilsResult_ProbVec = LogData_Start_ProbVec.signals.values;
        HilsResult_TimeIndexSt = LogData_Start_ProbVec.time;
        HilsResult_TimeIndexBrk = LogData_Stop_BrakingDataArray.time;
        HilsResult_TimeIndexEd = LogData_Start_ProbVec.time + LogData_Stop_BrakingState.signals.values(:,6)*0.01;

        HilsResult_MapArry_MaxAcc = LrnResult_EcpParamVec_MaxAcc.signals.values;
        HilsResult_MapArry_AprStall = LrnResult_EcpParamVec_AcrStall.signals.values;
        HilsResult_MapArry_TpShift = LrnResult_EcpParamVec_TpInit.signals.values;
        HilsResult_MapArry_TpDelta = LrnResult_EcpParamVec_TpDelta.signals.values;
        HilsResult_MapArry_AcdStall = LrnResult_EcpParamVec_AcdStall.signals.values;
        HilsResult_MapArry_AdjGain = LrnResult_EcpParamVec_AdjGain.signals.values;
        for i=1:IndexDataRange
            tmpAccRef = LogData_Stop_BrakingDataArray.signals.values(:,1,i);        
            tmpStTime = LogData_Start_ProbVec.time(i);
            tmpBrakingRange = LogData_Stop_PreSize.signals.values(i);        
            tmpAccPre = LogData_While_PreProf.signals.values(uint32(tmpStTime*100):uint32(tmpStTime*100) + tmpBrakingRange -1,1);
        %       tmpBrakingTime = LogData_Stop_BrakingState.signals.values(i,6);        
            HilsResult_RMSE(i,1) = sqrt(mean(tmpAccRef(1:tmpBrakingRange) - tmpAccPre).^2);
        end

        LrnNum_Hils = LrnNum.signals.values(end);
        clearvars tmp*        
        save(SaveFileName,'Hils*','VehData_*')
        clearvars Hils*
    end
end
%% Data anal
% load('IdmParam_Integ.mat')
% tmpStartIndex = uint32(HilsResult_TimeIndexSt*100);
% tmpVelCoast = VehData_vel(tmpStartIndex);
% 
% 
% tmpLrnParam_accMax = HilsResult_LrnParam(:,5);
% tmpSlcParam_accMax = HilsResult_EcpParam(:,5);
% figure()
% plot(tmpVelCoast,tmpLrnParam_accMax,'o');hold on;
% plot(tmpVelCoast,tmpSlcParam_accMax,'^');
% plot(BaseMap_MinAcc_Index,BaseMap_MaxAcc_Data);
% plot(BaseMap_MinAcc_Index,HilsResult_MapArry_MaxAcc(72,:))
% 
% figure()
% plot(BaseMap_MinAcc_Index,HilsResult_MapArry_MaxAcc(1:10:72,:)');hold on
% plot(BaseMap_MinAcc_Index,HilsResult_MapArry_MaxAcc(72,:),'linewidth',2)
% 
% 
% figure;plot(HIlsResult_EstProf(:,1:2),'DisplayName','HIlsResult_EstProf(:,1:2)')
% hold on
% plot(VehData_acc)


