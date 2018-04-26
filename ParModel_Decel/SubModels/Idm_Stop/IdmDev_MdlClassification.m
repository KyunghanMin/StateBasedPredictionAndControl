% IdmDev_MdlClassification.m
% Date: 17/12/20
% Description: Parameter identification
clear all; close all; clc
KH_ColorCode;
%% Driver configuration
BaseMap_CoastSpeed_Index = (5:5:40)';

ModelConfig.TimeGap = 0;
ModelConfig.StopDis = 0.5;
ModelConfig.Delta = 4;
ModelConfig.MinDis = 2;
ModelConfig.CoastGas = 0.01;
ModelConfig.CoastVel = 5;
ModelConfig.AccFiltNum = 25;
ModelConfig.AdjDis = -ModelConfig.MinDis;
ModelConfig.AdjDisGainDisAdj = 0.01;
ModelConfig.AdjDisGainTerm = 0.1;
ModelConfig.VelAdjFac = 2;
ModelConfig.InitBuffTime = 50;

ModelConfig.stState.Coast = 1;
ModelConfig.stState.Init = 2;
ModelConfig.stState.Veladj = 3;
ModelConfig.stState.Disadj = 4;
ModelConfig.stState.Term = 5;
ModelConfig.stState.End = 6;
ModelConfig.stStateSize = 6;

ModelConfig.stStateFlag.Coast = 1;
ModelConfig.stStateFlag.Init = 2;
ModelConfig.stStateFlag.VeladjInit = 3;
ModelConfig.stStateFlag.DisadjInit = 4;
ModelConfig.stStateFlag.TermInit = 5;
ModelConfig.stStateFlag.End = 6;


LrnConfig.DataIndexLength = length(BaseMap_CoastSpeed_Index);
LrnConfig.ExpUpdateGain = 0.5;
LrnConfig.AdjUpdateGain = 0.0001;
LrnConfig.AdjLimUp = 0.3;
LrnConfig.AdjLimLow = 0.005;
LrnConfig.AccFiltNum = 41;
LrnConfig.AdjAccDiff = 0.05;
LrnConfig.VelEffMin = 0.01;
LrnConfig.VelTerm = 0.05;
LrnConfig.CoastTime = 5;
LrnConfig.CoastTimeRange = 10;
LrnConfig.VelDelInitCriValue = 0.01;
LrnConfig.VelDifRatFac_Init = 1.7;
LrnConfig.ProbVecStdVel = 3;
LrnConfig.ProbVecStdAcc = 0.3;
LrnConfig.VelDeltaOff = 1;
LrnConfig.AcrCalcWindow = 15;
LrnConfig.LrnSwt = 0;
LrnConfig.TermBuff = 40;

OnlineConfig.Ts = 0.01;
OnlineConfig.PreStep = 1;
OnlineConfig.TermBuff = 1000;
OnlineConfig.CalcParamSize = 6;
OnlineConfig.CalcParamIndex.AccrStall = 1;
OnlineConfig.CalcParamIndex.VelStall = 2;
OnlineConfig.CalcParamIndex.DisStall = 3;
OnlineConfig.CalcParamIndex.AccdDisAdj = 4;
OnlineConfig.CalcParamIndex.VelRatio = 5;
OnlineConfig.CalcParamIndex.AccdStall = 6;

OnlineConfig.EpcParamSize = 6;
OnlineConfig.EpcParamIndex.TpInit = 1;
OnlineConfig.EpcParamIndex.TpStall = 2;
OnlineConfig.EpcParamIndex.AcdStall = 3;
OnlineConfig.EpcParamIndex.AcrStall = 4;
OnlineConfig.EpcParamIndex.AccMax = 5;
OnlineConfig.EpcParamIndex.AdjGain = 6;

OnlineConfig.VehStateSize = 5;
OnlineConfig.VehStateIndex.Brk = 1;
OnlineConfig.VehStateIndex.Aps = 2;
OnlineConfig.VehStateIndex.Acc = 3;
OnlineConfig.VehStateIndex.Vel = 4;
OnlineConfig.VehStateIndex.Dis = 5;

OnlineConfig.CoastFlag.None = 0;
OnlineConfig.CoastFlag.Start = 1;
OnlineConfig.CoastFlag.Stop = 2;

OnlineConfig.DriveState.Driving = 0;
OnlineConfig.DriveState.Braking = 1;
OnlineConfig.DriveState.Stop = 2;

OnlineConfig.StopCond.Dis = 0.5;
OnlineConfig.StopCond.Vel = 0.01;
OnlineConfig.StopCond.Time = 2;

OnlineConfig.StartCond.Vel = 5;

OnlineConfig.CoastCond.Vel = 5;
OnlineConfig.CoastCond.Gas = 5;

OnlineConfig.CoastingIndexSize = 2000;

if (exist('ModelConfig.mat') == 2)
    save('ModelConfig.mat','ModelConfig','OnlineConfig','LrnConfig','-append')    
else
    save('ModelConfig.mat','ModelConfig','OnlineConfig','LrnConfig')
end
    

%% Coasting determination and reference velocity calculation
DriverSet = {'KH','YK','GB'};
k = 1;  i = 1;
for k = 1:3
    FileName = ['Data_' char(DriverSet(k)) '.mat'];
    load(FileName)
    for i = 1:DataCoast.CoastNum
%         for i = 4
        eval(['tmpData = DataCoast.Case' num2str(i) ';'])        
        CaseDataAps = tmpData(:,8);
        CaseDataBrk = tmpData(:,9);
        CaseDataVel = tmpData(:,4);
        CaseDataAcc = tmpData(:,1);
        CaseDataTime = (0.01:0.01:length(CaseDataAps)*0.01)';
        tmpAccMin = min(CaseDataAcc);
        tmpTp = find((CaseDataVel<LrnConfig.VelTerm) & (CaseDataTime>LrnConfig.CoastTime),1);
        if isempty(tmpTp)
            [tmpDummy tmpTp] = min(CaseDataVel);
        end

        CaseDataDis = tmpData(tmpTp+5,5) - tmpData(:,5);

        % Coasting data and reference velocity calculation       
        tmpTpBrk = find(CaseDataBrk>2,1);

        tmpDataVeh = [CaseDataAcc CaseDataVel CaseDataDis];

        [CalcData_VelRef CalcData_DisTerm CalcData_VelTerm] = Fcn_RefVelCalc(ModelConfig, LrnConfig, tmpDataVeh, tmpAccMin);         

        CalcData_AccRef = -0.5*CaseDataVel.^2./CaseDataDis;
        CalcData_AccRef(tmpTp:end) = CalcData_AccRef(tmpTp);

        CalcData_VelRefDelta = CalcData_VelRef(2:end) - CalcData_VelRef(1:end-1);    
        CalcData_VelDelta = CaseDataVel(2:end,:) - CaseDataVel(1:end-1,:);
        CalcData_VelDiff = CaseDataVel - CalcData_VelRef;
        CalcData_VelDiffFilt = sgolayfilt(CalcData_VelDiff,1,ModelConfig.AccFiltNum);
        
        tmpCaseDataAcc = CaseDataAcc; tmpCaseDataAcc(tmpCaseDataAcc>=0) = 0;
        CalcData_DisEff = (tmpCaseDataAcc./-6).^0.5.*CaseDataDis;

        tmpData(:,11) = CalcData_VelRef;
        tmpData(:,12) = CalcData_AccRef;
        tmpData(:,13) = CalcData_VelDiffFilt;
        tmpData(:,14) = CaseDataDis;
        tmpData(:,15) = CalcData_DisEff;
        
        veh_vel_calc = cumtrapz(CaseDataAcc)*0.01;
        [tmpDummy tmpStopPoint] = min(veh_vel_calc);
        veh_vel_calc = veh_vel_calc-tmpDummy; 
        tmpData(:,16) = veh_vel_calc;
        veh_dis_calc = cumtrapz(veh_vel_calc)*0.01;        
        veh_dis_calc = veh_dis_calc(tmpStopPoint) - veh_dis_calc;
        tmpData(:,17) = veh_dis_calc;
        
        CalcData_AccRefCalc = -0.5*veh_vel_calc.^2./veh_dis_calc;
        CalcData_AccRefCalc(tmpTp:end) = CalcData_AccRefCalc(tmpTp);    
        
        tmpData(:,18) = CalcData_AccRefCalc;
        Param.Ap_Ref_Coast(i) = CalcData_AccRef(1);        
        Param.Ap_Coast(i) = CaseDataAcc(1);        
        Param.ApFact(i) = Param.Ap_Coast(i) - Param.Ap_Ref_Coast(i);
        Param.Tp_Shift(i) = tmpTpBrk;
        Param.Tp_Term(i) = tmpTp;
        tmpVelDiffInit = mean(CalcData_VelDiffFilt(1:40));
        tmpTpInit = find(CalcData_VelDiffFilt(1:tmpTp)>=tmpVelDiffInit*LrnConfig.VelDifRatFac_Init,1);  
        if isempty(tmpTpInit)
            tmpTpInit = tmpTpBrk;
        end        
        Param.Tp_Init(i) = tmpTpInit;
        tmpVpDiffInit = CalcData_VelDiffFilt(tmpTpInit);
        [tmpDummy tmpTpMax] = max(CalcData_VelDiffFilt(tmpTpBrk:Param.Tp_Term(i)));
        
        tmpTpDelta = find(CalcData_VelDiffFilt(tmpTpBrk:tmpTp)>=tmpVpDiffInit+LrnConfig.VelDeltaOff,1);        
        if isempty(tmpTpDelta)
            tmpTpDelta = tmpTpMax;
        end         
        tmpTpDelta = tmpTpDelta-1;        
        tmpTpStall = tmpTpMax + tmpTpBrk - 1;
        
        Param.Tp_Delta(i) = tmpTpDelta;
        Param.Tp_Stall(i) = tmpTpStall;

        Param.Vp_Coast(i) = CaseDataVel(1);
        Param.Vp_Stall(i) = CalcData_VelRef(Param.Tp_Stall(i));
        Param.Vp_Delta(i) = CalcData_VelRef(Param.Tp_Delta(i));
        
        Param.Delta(i) = (CalcData_VelRef(Param.Tp_Delta(i)) - CalcData_VelRef(Param.Tp_Init(i)))/(Param.Tp_Delta(i) - Param.Tp_Init(i));
                
        tmpVelDifRatFac = CalcData_VelDiffFilt(tmpTpBrk)/CalcData_VelDiffFilt(1);
        
        Param.VpdiffRat(i) = tmpVelDifRatFac;
        tmpAppStall = mean(CaseDataAcc(tmpTpStall-LrnConfig.AcrCalcWindow:tmpTpStall+LrnConfig.AcrCalcWindow));
        tmpAppRefStall = CalcData_AccRefCalc(tmpTpStall);        
        tmpAppRefInit =  CalcData_AccRefCalc(tmpTpBrk);
        tmpAppInit = CaseDataAcc(tmpTpBrk);
        Param.Aprat_Stall(i) = tmpAppStall/tmpAppRefStall;        
        Param.Apdiff_Stall(i) = (tmpAppRefStall-tmpAppRefInit);
        Param.ApDelta(i) = (tmpAppStall - tmpAppInit)/tmpTpDelta;
        Param.Ap_Min(i) = tmpAccMin;        
        Param.Dp_Coast(i) = veh_dis_calc(1);                
        Param.ApFact_Init(i) = tmpAppInit - tmpAppRefInit;
        tmpData(:,19) = CaseDataTime;
        
        eval(['DataCoast.Case' num2str(i) ' = tmpData;']);
    end
    eval(['Param_' char(DriverSet(k)) ' = Param;']);
    save(FileName,'DataCoast','Param','-append')
    clearvars Param;    
end
save('IdmParam.mat','Param_*');
%% Parameter plot
figure
for k = 1:3
    FileName = ['Data_' char(DriverSet(k)) '.mat'];
    load(FileName)
       
    for i = 1:10
        eval(['tmpData = DataCoast.Case' num2str(i) ';'])
        CaseDataAps = tmpData(:,8);
        CaseDataBrk = tmpData(:,9);
        CaseDataVel = tmpData(:,4);
        CaseDataAcc = tmpData(:,1);
        CaseDataVelRef = tmpData(:,11);
        CaseDataAccRef = tmpData(:,12);
        CaseDataVelDiff = tmpData(:,13);
        CaseDataDis = tmpData(:,14);
        plot(CaseDataVelDiff);hold on;
    end
end

figure
for i = 1:5
        eval(['tmpData = DataCoast.Case' num2str(i) ';'])
        CaseDataAps = tmpData(:,8);
        CaseDataBrk = tmpData(:,9);
        CaseDataVel = tmpData(:,4);
        CaseDataAcc = tmpData(:,1);
        CaseDataVelRef = tmpData(:,11);
        CaseDataAccRef = tmpData(:,12);
        CaseDataVelDiff = tmpData(:,13);
        CaseDataDis = tmpData(:,14);
        subplot(2,1,1)
        plot(CaseDataVelDiff,'color',Color.SP(i,:));hold on;
        plot(Param.Tp_Stall(i),CaseDataVelDiff(Param.Tp_Stall(i)),'o','color',Color.SP(i,:));hold on;
        plot(Param.Tp_Init(i),CaseDataVelDiff(Param.Tp_Init(i)),'^','color',Color.SP(i,:));hold on;
        plot(Param.Tp_Delta(i),CaseDataVelDiff(Param.Tp_Delta(i)),'sq','color',Color.SP(i,:));hold on;
        subplot(2,1,2)
        plot(CaseDataAcc,'color',Color.SP(i,:));hold on;
        plot(CaseDataAccRef,'--','color',Color.SP(i,:));hold on;
        plot(Param.Tp_Stall(i),CaseDataAcc(Param.Tp_Stall(i)),'o','color',Color.SP(i,:));hold on;        
end
%% Base parameter calculation
clear all;
load('IdmParam.mat')
clearvars -except Param_*
Data_TpShift = [Param_KH.Tp_Shift Param_YK.Tp_Shift Param_GB.Tp_Shift];
Data_TpInit = [Param_KH.Tp_Init Param_YK.Tp_Init Param_GB.Tp_Init];
Data_TpStall = [Param_KH.Tp_Stall Param_YK.Tp_Stall Param_GB.Tp_Stall];
Data_TpMax = Data_TpStall-Data_TpShift;
Data_TpDelta = [Param_KH.Tp_Delta Param_YK.Tp_Delta Param_GB.Tp_Delta];
Data_AprStall = [Param_KH.Aprat_Stall Param_YK.Aprat_Stall Param_GB.Aprat_Stall];
Data_VpStall = [Param_KH.Vp_Stall Param_YK.Vp_Stall Param_GB.Vp_Stall];
Data_MinAcc = [Param_KH.Ap_Min Param_YK.Ap_Min Param_GB.Ap_Min];
Data_Index_CoastSpeed = [Param_KH.Vp_Coast Param_YK.Vp_Coast Param_GB.Vp_Coast];
Data_Index_CoastAcDiff = [Param_KH.ApFact Param_YK.ApFact Param_GB.ApFact];
Data_Index_CoastAcDiffInit = [Param_KH.ApFact_Init Param_YK.ApFact_Init Param_GB.ApFact_Init];
Data_AccRefDiff = [Param_KH.Apdiff_Stall Param_YK.Apdiff_Stall Param_GB.Apdiff_Stall];
Data_AccDelta = [Param_KH.ApDelta Param_YK.ApDelta Param_GB.ApDelta];

% Map generation
BaseMap_CoastSpeed_Index = (5:5:40)';
BaseMap_CoastAccdiff_Index = (0:0.5:3.5)';

ft = fittype( 'poly1' );
% Mean value 
% - Acc ratio
[fitresult, gof] = fit(Data_Index_CoastSpeed', Data_AprStall', ft );
BaseMap_AprStall_Data = BaseMap_CoastSpeed_Index;
BaseMap_AprStall_Data = fitresult(BaseMap_AprStall_Data);

% Calculation value for coast acc diff
% - Acc Delta
ft = fittype( 'exp1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [-0.00261343403031386 1.18333383374591];
[fitresult, gof] = fit( Data_Index_CoastAcDiffInit', Data_AccDelta', ft, opts );
BaseMap_AccDelta_Index = BaseMap_CoastAccdiff_Index;
BaseMap_AccDelta_Data = fitresult(BaseMap_AccDelta_Index);

% - Acc Diff
ft = fittype( 'exp1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Robust = 'LAR';
opts.Display = 'Off';
opts.StartPoint = [-0.159819581970119 0.619753729688331];
[fitresult, gof] = fit( Data_Index_CoastAcDiffInit', Data_AccRefDiff', ft, opts );
BaseMap_AccDiff_Index = BaseMap_CoastAccdiff_Index;
BaseMap_AccDiff_Data = fitresult(BaseMap_AccDiff_Index);


% Calculation value for coast acc diff
% - Tp init
% - Tp delta
ft = fittype( 'exp1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [400 -1];
[fitresult, gof] = fit( Data_Index_CoastAcDiff', Data_TpShift', ft, opts );
BaseMap_TpShift_Index = BaseMap_CoastAccdiff_Index;
BaseMap_TpShift_Data = fitresult(BaseMap_TpShift_Index);

opts.StartPoint = [237 -0.1];
[fitresult, gof] = fit(Data_Index_CoastAcDiff', Data_TpMax', ft );
BaseMap_TpDelta_Index = BaseMap_CoastAccdiff_Index;
BaseMap_TpDelta_Data = fitresult(BaseMap_TpDelta_Index);

% Calculation value for coast speed
[fitresult, gof] = fit(Data_Index_CoastSpeed', Data_VpStall', ft );
BaseMap_VpStall_Index = BaseMap_CoastSpeed_Index;
BaseMap_VpStall_Data = fitresult(BaseMap_VpStall_Index);


[fitresult, gof] = fit(Data_Index_CoastSpeed', Data_MinAcc', ft );
BaseMap_MinAcc_Index = BaseMap_CoastSpeed_Index;
BaseMap_MinAcc_Data = fitresult(BaseMap_MinAcc_Index);

BaseMap_MaxAcc_Data = -BaseMap_MinAcc_Data;
BaseMap_AdjGain_Data = 0.01*ones(8,1);

if (exist('IdmParam.mat') == 2)
    save('IdmParam.mat','BaseMap_*','-append');
else
    save('IdmParam.mat','BaseMap_*');
end

if (exist('IdmParam_Integ.mat') == 2)
    save('IdmParam_Integ.mat','BaseMap_*','-append');
else
    save('IdmParam_Integ.mat','BaseMap_*');
end    