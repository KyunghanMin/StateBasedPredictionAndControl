% IdmDev_MdlOnlineImplement.m
% 18/04/12
clear all;close all;clc
load('Data_GB.mat')
load('ModelConfig.mat')
load('IdmParam.mat')
load('IdmParam_Integ.mat')
clearvars DataCase_*
%% MC learning and TD implementation
clearvars tmp*
clearvars ModProf_*
clearvars ArryModProf_*
clearvars LrnParam*

BrakeCri = 0.05;
Index_ModLength = 1500;
%% Sim
tic
% Simulation case
%   - Repeat 10 times
Index_CaseSet = 1*ones(10,1);

Indexcase = 1;

% Initialize learning parameter vector
%       Stall = Adjustment 
LrnParamVec_AccMax = BaseMap_MaxAcc_Data;
LrnParamVec_AprStall = BaseMap_AprStall_Data;
LrnParamVec_TpInit = BaseMap_TpShift_Data;
LrnParamVec_TpDelta = BaseMap_TpDelta_Data;
LrnParamVec_AdjGain = BaseMap_AdjGain_Data;
LrnParamVec_AccDelta = BaseMap_AccDelta_Data;
LrnParamVec_ApdStall = BaseMap_AccDiff_Data;

SimSwitchParamLearn = 0;
% 0: Learning parameter
% 1: Determination parameter
% 2: Custom parameter
    % ParamSel.AccDiff = -0.6181;
    % ParamSel.Tp_Delta = 319;
    % AccDiffSet = [ParamSel.AccDiff ParamSel.AccDiff*0.8 ParamSel.AccDiff*1.2];
    % TpDeltaSet = [ParamSel.Tp_Delta ParamSel.Tp_Delta*0.8 ParamSel.Tp_Delta*1.2];
    % AdjGainSet = [0.022 0.022*0.7 0.022*1.5];
%% Iteration whole case
for Indexcase = 1:length(Index_CaseSet)   
    %% Parameter activation
    % Import case data
    Index_case = Index_CaseSet(Indexcase);                
    eval(['tmpData = DataCoast.Case' num2str(Index_case) ';']);
    % Data arrangement - Vehicle
    %   - Data_Cur = Acceleration, Velocity, Distance
    Data_Cur = [tmpData(:,1) tmpData(:,4) tmpData(:,14)];
    Data_Cur(:,3) = Data_Cur(:,3) + ModelConfig.StopDis;   
    %   - Parameter of case
    ParamCase = Fcn_ParamSelect(Param,Index_case);       
    VehData_acc = Data_Cur(:,1);
    VehData_acc(ParamCase.Tp_Term:Index_ModLength) = VehData_acc(ParamCase.Tp_Term);
    VehData_vel = Data_Cur(:,2);
    VehData_vel(ParamCase.Tp_Term:Index_ModLength) = VehData_vel(ParamCase.Tp_Term);    
    VehData_dis = Data_Cur(:,3);
    VehData_dis(ParamCase.Tp_Term:Index_ModLength) = VehData_dis(ParamCase.Tp_Term);
    VehData_time = (OnlineConfig.Ts:OnlineConfig.Ts:length(VehData_acc)*OnlineConfig.Ts)';
    VehData_brk = tmpData(:,9);
    %    - Coasting index = Coast velocity, Coast acceleration diff
    VehData_CoastVel = VehData_vel(1);
    VehData_AccFct = VehData_acc(1) + 0.5*VehData_CoastVel^2/VehData_dis(1);        
    % Learning parameter import    
    %    - Coast velocity indexing parameters: MaxAcc, AdjGain
    [ModelParam.MaxAcc ModelParam.IndexProbVec_Vel] = Fcn_FzBlending(VehData_CoastVel,BaseMap_CoastSpeed_Index,LrnParamVec_AccMax);        
    ModelParam.AdjGain = Fcn_FzBlending(VehData_CoastVel,BaseMap_CoastSpeed_Index,LrnParamVec_AdjGain);
    ModelParam.AdjGain(ModelParam.AdjGain>=LrnConfig.AdjLimUp) = LrnConfig.AdjLimUp;
    ModelParam.AdjGain(ModelParam.AdjGain<=LrnConfig.AdjLimLow) = LrnConfig.AdjLimLow;    
    % Learning parameter import    
    %    - Coast acceleration diff indexing parameters: TpShift(Init),
    %    TpStall, AprStall, ApdStall
    [ModelParam.AprStall ModelParam.IndexProbVec_Acc]= Fcn_FzBlending(VehData_AccFct,BaseMap_CoastAccdiff_Index,LrnParamVec_AprStall);
    ModelParam.TpDelta = Fcn_FzBlending(VehData_AccFct,BaseMap_CoastAccdiff_Index,LrnParamVec_TpDelta);    
    ModelParam.TpInit = Fcn_FzBlending(VehData_AccFct,BaseMap_CoastAccdiff_Index,LrnParamVec_TpInit);

    
    ModelParam.SlcParamArry(Indexcase,1) = ModelParam.MaxAcc;
    ModelParam.SlcParamArry(Indexcase,2) = ModelParam.AprStall;
    ModelParam.SlcParamArry(Indexcase,3) = ModelParam.TpInit;
    ModelParam.SlcParamArry(Indexcase,4) = ModelParam.TpDelta;
    ModelParam.SlcParamArry(Indexcase,6) = ModelParam.AdjGain;


    
    % Applying determination parameters
    if SimSwitchParamLearn == 1
        ParamSel = ParamCase;   
        ParamSel.Tp_Init = ParamCase.Tp_Shift;
        ParamSel.MaxAcc = ModelParam.MaxAcc;
        ParamSel.AdjGain = ModelParam.AdjGain;
    else
        % Parameter activation using learning parameter
        ParamSel.Aprat_Stall = ModelParam.AprStall;
        ParamSel.Ap_Min = -ModelParam.MaxAcc;
        ParamSel.Tp_Shift = ModelParam.TpInit;   
        ParamSel.Tp_Init = ModelParam.TpInit;
        ParamSel.Tp_Delta = ModelParam.TpDelta;
        ParamSel.Tp_Term = ParamCase.Tp_Term;
        ParamSel.Vp_Coast = VehData_CoastVel;
        ParamSel.AdjGain = ModelParam.AdjGain;
        ParamSel.MaxAcc = ModelParam.MaxAcc;    
    end
    
    
    % State flag
    Flag_Coast = 0;
    Flag_Init = 0;
    Flag_Stall = 0;        
    Flag_Veladj = 0;
    Flag_VeladjInit = 0;
    Flag_Disadj = 0;
    Flag_DisadjInit = 0;
    Flag_Term = 0;
    Flag_TermInit = 0;
    Flag_StopCond = 0;
    % Model profiles - Global
    ModProf_dis_adj = 0*(1:Index_ModLength)';
    ModProf_dis_del = 0*(1:Index_ModLength)';
    ModProf_acc_ref = 0*(1:Index_ModLength)';
    ModProf_state = ones(Index_ModLength,1);
    ModProf_acc_est =  0*(1:Index_ModLength)';
    ModProf_vel_est =  0*(1:Index_ModLength)';
    ModProf_dis_est =  0*(1:Index_ModLength)';
    ModProf_vel_ref =  0*(1:Index_ModLength)';
    % Model parameter - Global
    ModelParam.acc_stall = 0;   
    ModelParam.vel_stall = 0;
    ModelParam.dis_stall = 0;        
    ModelParam.accdif_disadj = 0;
    ModelParam.acc_disadj_ref =0;
    ModelParam.acc_disadj = 0;
    ModelParam.accr_stall = 0;
    ModelParam.velr_stall = 0;        
    ModelParam.disr_term = 0;
    
    % Apply prediction step
    % ----- MonteCarlo prediction: Predict deceleration at first time step only
    PreIndexSet = 1;
    PreIndex = PreIndexSet(1);k = 1;
    Index_timestep = 1;   
    
    %% Acc estimation - Time iteration
    % ----- Temporary time frame work for TD or advanced algorithm
    % ----- For braking scenario prediction occurs only 1st time step
    for Index_timestep = 1:ParamSel.Tp_Term
        % Local estimation profiles 
        % Index_timestep
        tmpLoc_Mod_vel_est = 0*(1:Index_ModLength)';
        tmpLoc_Mod_dis_est = 0*(1:Index_ModLength)';
        tmpLoc_Mod_acc_est = 0*(1:Index_ModLength)';
        tmpLoc_Mod_acc_est_veff = 0*(1:Index_ModLength)';
        tmpLoc_Mod_acc_est_deff = 0*(1:Index_ModLength)';
        tmpLoc_ModProf_dis_adj = 0*(1:Index_ModLength)';
        tmpLoc_ModProf_dis_del = 0*(1:Index_ModLength)';
        tmpLoc_ModProf_acc_ref = 0*(1:Index_ModLength)';
        tmpLoc_ModProf_vel_ref = 0*(1:Index_ModLength)';

        tmpStateArry = 0*(1:Index_ModLength)';
        % Arrangement of current step data and state;            
        Data_CurStep = Data_Cur(Index_timestep,:);
        tmpLoc_ModProf_dis_adj(Index_timestep:Index_ModLength) = ModProf_dis_adj(Index_timestep:Index_ModLength);
        tmpLoc_ModProf_dis_del(Index_timestep:Index_ModLength) = ModProf_dis_del(Index_timestep:Index_ModLength);
        tmpLoc_ModProf_acc_ref(Index_timestep:Index_ModLength) = ModProf_acc_ref(Index_timestep:Index_ModLength);
        tmpLoc_ModProf_vel_ref(Index_timestep:Index_ModLength) = ModProf_vel_ref(Index_timestep:Index_ModLength);

        % Estimation profile initilaization             
        tmpLoc_Mod_acc_est(Index_timestep) = Data_CurStep(1);
        tmpLoc_Mod_vel_est(Index_timestep) = Data_CurStep(2);
        tmpLoc_Mod_dis_est(Index_timestep) = Data_CurStep(3);  

        tmpVelRefInit = tmpLoc_Mod_vel_est(Index_timestep)./(1 - tmpLoc_Mod_acc_est(Index_timestep)/ParamSel.MaxAcc)^0.25;
        tmpLoc_ModProf_vel_ref(Index_timestep)= tmpVelRefInit;
        % Parameter initialization           
        tmpLoc_ModelParam_accdiff = ModelParam.accdif_disadj;
        tmpLoc_ModelParam_accr_stall = ModelParam.accr_stall;
        tmpLoc_ModelParam_velr_stall = ModelParam.velr_stall;
        tmpLoc_ModelParam_disr_term = ModelParam.disr_term;
        tmpLoc_ModelParam_dis_stall = ModelParam.dis_stall;
        tmpLoc_ModelParam_vel_stall = ModelParam.vel_stall;

        tmpLocFlag_VeladjInit = Flag_VeladjInit;
        tmpLocFlag_DisadjInit = Flag_DisadjInit;
        tmpLocFlag_TermInit = Flag_TermInit;
        tmpLocFlag_StopCond = Flag_StopCond;
        tmpLocFlag_PreStart = 0;
        tmpLocFlag_RefVelInit = 0;
        
        tmpLocParam_VelRefDiff = 0;
        tmpLocParam_VelRefInit = 0;
        tmpModProf = 0;
        
        tmpLocParam_AccDiffIndex = 0;
        tic;
        % Set the prediction step
        if  Index_timestep == PreIndex
            indexlen_prestep = (ParamSel.Tp_Term + OnlineConfig.TermBuff) - Index_timestep;                                
        else
            indexlen_prestep = 1;
        end
        tmpLocFlag_PreState = ModProf_state(Index_timestep);
        index_prestep = 1;
       %% prediction iteration
        for index_prestep = 1:indexlen_prestep
            index_curstep = Index_timestep + index_prestep - 1;
            % Initilization of profile data
            %  Model profile vector: AccEst, AccRef, VelEst, VelRef,
            %                        DisEst, DisAdj, DisDel, 
            %                        (DisEff = DisAdj + DisDel)
            tmpModProf = [tmpLoc_Mod_acc_est(index_curstep) tmpLoc_ModProf_acc_ref(index_curstep) ...
            tmpLoc_Mod_vel_est(index_curstep) tmpLoc_ModProf_vel_ref(index_curstep) ...
            tmpLoc_Mod_dis_est(index_curstep) tmpLoc_ModProf_dis_adj(index_curstep) tmpLoc_ModProf_dis_del(index_curstep)];                
            % State definition for each prediction step
            stState = Fcn_StateEstimation(index_curstep, ParamSel, tmpModProf, tmpLocFlag_StopCond, tmpLocFlag_PreState);                     
            
            %% Determination of estimated model profiles for each step
            %  AccRef --> VelRef --> DisAdj --> DisDel            
            switch stState

                % Coasting section
                case ModelConfig.stState.Coast
                    tmpLoc_ModProf_acc_ref(index_curstep+1) = -0.5*tmpLoc_Mod_vel_est(index_curstep).^2/(tmpLoc_Mod_dis_est(index_curstep) - ModelConfig.StopDis);
                    tmpLoc_ModProf_vel_ref(index_curstep+1) = tmpLoc_ModProf_vel_ref(index_curstep) + tmpLoc_Mod_acc_est(index_curstep)*0.01;
                    tmpLoc_ModProf_dis_adj(index_curstep+1) = 0;
                    tmpLoc_ModProf_dis_del(index_curstep+1) = 0;
                    
                % Initial section
                case ModelConfig.stState.Init
                    tmpLoc_ModProf_acc_ref(index_curstep+1) = -0.5*tmpLoc_Mod_vel_est(index_curstep).^2/(tmpLoc_Mod_dis_est(index_curstep) - ModelConfig.StopDis);   
                    if tmpLocFlag_RefVelInit == 0
                        % At initial point of initial section
                        %  --- Calculate reference velocity ratio 
                        %  --- VelRef = VelEst*VelRatio
                        %  --- VelRatio = f(AccSlope)
                        %  --- AccSlopeInit = f(ApdStall, AcrStall,TpDelta)
                        tmpLocFlag_RefVelInit = 1;
                        tmpLocParam_VelRefInit = tmpLoc_Mod_vel_est(index_curstep)./(1 - tmpLoc_Mod_acc_est(index_curstep)/ParamSel.MaxAcc)^0.25;
                        tmpLocParam_VelRatioInit = tmpLoc_Mod_vel_est(index_curstep)/tmpLocParam_VelRefInit;
                        tmpLoc_ModProf_vel_ref(index_curstep+1) = tmpLocParam_VelRefInit;
                        tmpLocParam_AccDiffIndex = tmpLoc_Mod_acc_est(index_curstep) - tmpLoc_ModProf_acc_ref(index_curstep+1);
                        % Activate ApdStall
                        if SimSwitchParamLearn == 1 
                            tmpLocParam_AccDiff = ParamSel.AccDiff;
                            ModelParam.ApdStall = tmpLocParam_AccDiff;
                            ModelParam.SlcParamArry(Indexcase,5) = ModelParam.ApdStall;
                        else
                            % For learning case ApdStall is activated according to acceleration diff index at initial point
                            tmpLocParam_AccDiff = Fcn_FzBlending(tmpLocParam_AccDiffIndex,BaseMap_CoastAccdiff_Index,LrnParamVec_ApdStall); 
                            ModelParam.ApdStall = tmpLocParam_AccDiff;
                            ModelParam.SlcParamArry(Indexcase,5) = ModelParam.ApdStall;
                        end                        
                        tmpLocParam_AccEstSet = (tmpLocParam_AccDiff +tmpLoc_ModProf_acc_ref(index_curstep))*ParamSel.Aprat_Stall;                         
                        tmpLocParam_AccDelta = (tmpLocParam_AccEstSet - tmpLoc_Mod_acc_est(index_curstep))/ParamSel.Tp_Delta;
                        tmpLocParam_VelRatio = tmpLocParam_VelRatioInit;
                    else
                        tmpLocParam_VelRatio = (tmpLocParam_VelRatio^ModelConfig.Delta - tmpLocParam_AccDelta/ParamSel.MaxAcc)^(1/ModelConfig.Delta);
                        tmpLoc_ModProf_vel_ref(index_curstep+1) = tmpLoc_Mod_vel_est(index_curstep)/tmpLocParam_VelRatio;
                    end
                    tmpLoc_ModProf_dis_adj(index_curstep+1) = 0;
                    tmpLoc_ModProf_dis_del(index_curstep+1) = 0;
                    
                % Adjustment section - velocity adjust
                %  ---- Adjust VelRef to DisAdj == DisEff
                case ModelConfig.stState.Veladj
                    tmpLoc_ModProf_acc_ref(index_curstep+1) = -0.5*tmpLoc_Mod_vel_est(index_curstep).^2/(tmpLoc_Mod_dis_est(index_curstep) - ModelConfig.StopDis);                        
                    if (tmpLocFlag_VeladjInit == 0) && (Flag_VeladjInit == 0) % Store prediction info at adjustment start
                        tmpLoc_ModelParam_vel_stall = tmpLoc_Mod_vel_est(index_curstep);
                        tmpLoc_ModelParam_dis_stall = tmpLoc_Mod_dis_est(index_curstep);
                        tmpLoc_ModelParam_accr_stall = tmpLoc_Mod_acc_est(index_curstep)./ParamSel.MaxAcc;
                        tmpLoc_ModelParam_velr_stall = (1 - tmpLoc_ModelParam_accr_stall);
                        tmpLoc_ModelParam_velr_stall(tmpLoc_ModelParam_velr_stall<=0) = 0;                            
                        tmpLoc_ModProf_dis_adj(index_curstep+1) = 0;                                                    
                        tmpLocFlag_VeladjInit = 1;                                                        
                    else % Update value of DisAdj using above info                                      
                        tmpLoc_ModProf_dis_adj(index_curstep+1) = (tmpLoc_ModelParam_dis_stall - tmpLoc_Mod_dis_est(index_curstep))*tmpLoc_ModelParam_vel_stall/ModelConfig.VelAdjFac;                            
                    end
                    tmpCalcDem = (tmpLoc_ModelParam_velr_stall - (tmpLoc_ModProf_dis_adj(index_curstep+1)/tmpLoc_Mod_dis_est(index_curstep))^2);
                    tmpCalcDem(tmpCalcDem<=0) = 0.00001;
                    tmpLoc_ModProf_vel_ref(index_curstep+1) = tmpLoc_Mod_vel_est(index_curstep) / tmpCalcDem^(1/ModelConfig.Delta);
                    tmpLoc_ModelParam_accdiff = (tmpLoc_Mod_acc_est(index_curstep) - tmpLoc_ModProf_acc_ref(index_curstep));
                    tmpLoc_ModProf_dis_del(index_curstep+1) = 0;
                    
                % Adjustment section - distance adjust
                %  ---- Adjust DisDel to AccEst == AccRef
                case ModelConfig.stState.Disadj
                    tmpLoc_ModProf_acc_ref(index_curstep+1) = -0.5*tmpLoc_Mod_vel_est(index_curstep).^2/(tmpLoc_Mod_dis_est(index_curstep) - ModelConfig.StopDis);
                    tmpLoc_ModProf_dis_adj(index_curstep+1) = tmpLoc_Mod_dis_est(index_curstep);
                    tmpCalcDem = (tmpLoc_ModelParam_velr_stall - (tmpLoc_ModProf_dis_adj(index_curstep+1)/tmpLoc_Mod_dis_est(index_curstep))^2);
                    tmpCalcDem(tmpCalcDem<=0) = 0.00001;                    
                    tmpLoc_ModProf_vel_ref(index_curstep+1) = tmpLoc_Mod_vel_est(index_curstep) / tmpCalcDem^(1/ModelConfig.Delta);  
                    if (tmpLocFlag_DisadjInit == 0) && (Flag_DisadjInit == 0)                           
                        ModelParam.accdif_disadj = 0;
                        ModelParam.acc_disadj_ref =0;
                        ModelParam.acc_disadj = 0;        
                        tmpLoc_ModelParam_accdiff = (tmpLoc_Mod_acc_est(index_curstep) - tmpLoc_ModProf_acc_ref(index_curstep));
                        % State assign the sign of AccRef - AccEst to convergence
                        %    - StopCond = 1 -> Start Termination when AccEst < AccRef
                        %    - StopCond = -1 -> Start Termination when AccEst > AccRef
                        if tmpLoc_ModelParam_accdiff >= 0  
                            tmpLocFlag_StopCond = 1; 
                        else 
                            tmpLocFlag_StopCond = -1;
                        end
                        tmpLoc_ModProf_dis_del(index_curstep+1) = 0;
                        tmpLocFlag_DisadjInit = 1;
                        tmpTpAdj = index_curstep;
                    else
                        tmpLoc_ModProf_dis_del(index_curstep+1) = tmpLoc_ModProf_dis_del(index_curstep) + tmpLoc_ModelParam_accdiff*ParamSel.AdjGain;
                    end

                % Adjustment section - distance adjust
                %  ---- Adjust DisDel to AccEst == AccRef
                case ModelConfig.stState.Term
                    tmpLoc_ModProf_acc_ref(index_curstep+1) = -0.5*tmpLoc_Mod_vel_est(index_curstep).^2/(tmpLoc_Mod_dis_est(index_curstep) - ModelConfig.StopDis);
                    tmpLoc_ModProf_dis_adj(index_curstep+1) = tmpLoc_Mod_dis_est(index_curstep);
                    tmpCalcDem = (tmpLoc_ModelParam_velr_stall - (tmpLoc_ModProf_dis_adj(index_curstep+1)/tmpLoc_Mod_dis_est(index_curstep))^2);
                    tmpCalcDem(tmpCalcDem<=0) = 0.00001;                    
                    tmpLoc_ModProf_vel_ref(index_curstep+1) = tmpLoc_Mod_vel_est(index_curstep) / tmpCalcDem^(1/ModelConfig.Delta);                        
                    if (tmpLocFlag_TermInit == 0) && (Flag_TermInit == 0)                     
                        tmpLoc_ModelParam_disr_term = (tmpLoc_ModProf_dis_adj(index_curstep)+tmpLoc_ModProf_dis_del(index_curstep))/tmpLoc_ModProf_dis_adj(index_curstep);
                        tmpLocFlag_TermInit = 1;
                        tmpTpTerm = index_curstep;
                    end
                    tmpLoc_ModProf_dis_del(index_curstep+1) = tmpLoc_ModProf_dis_del(index_curstep) + ModelConfig.AdjDisGainTerm*(tmpLoc_Mod_acc_est(index_curstep) - tmpLoc_ModProf_acc_ref(index_curstep));
            end
            %% Profile update
            tmpModProf = [tmpLoc_Mod_acc_est(index_curstep) tmpLoc_ModProf_acc_ref(index_curstep+1) ...
            tmpLoc_Mod_vel_est(index_curstep) tmpLoc_ModProf_vel_ref(index_curstep+1) ...
            tmpLoc_Mod_dis_est(index_curstep) tmpLoc_ModProf_dis_adj(index_curstep+1) tmpLoc_ModProf_dis_del(index_curstep+1)];
            % Acceleration calculation
            [tmpLoc_Mod_acc_est(index_curstep+1) tmpLoc_Mod_acc_est_veff(index_curstep+1) tmpLoc_Mod_acc_est_deff(index_curstep+1)] = Fcn_AccCalc(tmpModProf,ModelConfig,ParamSel);
            % State update
            tmpLoc_Mod_vel_est(index_curstep+1) = tmpLoc_Mod_vel_est(index_curstep) + tmpLoc_Mod_acc_est(index_curstep+1)*OnlineConfig.Ts;
            if tmpLoc_Mod_vel_est(index_curstep+1) <= 0.01
                tmpLoc_Mod_vel_est(index_curstep+1) = 0.01;                
                tmpLoc_Mod_dis_est(index_curstep+1) = tmpLoc_Mod_dis_est(index_curstep);
            else                   
                tmpLoc_Mod_dis_est(index_curstep+1) = tmpLoc_Mod_dis_est(index_curstep) - tmpLoc_Mod_vel_est(index_curstep+1)*OnlineConfig.Ts;
            end
            
            if tmpLoc_Mod_dis_est(index_curstep+1) <= ModelConfig.StopDis
                tmpLoc_Mod_dis_est(index_curstep+1) = ModelConfig.StopDis;
            end

            % State matching = (State, Flag of 1st iteration means the global truth);
            if tmpLocFlag_PreStart == 0
            % Global stat, parameter and profile update for TD estimation
                tmpLocFlag_PreStart = 1;
                Flag_VeladjInit = tmpLocFlag_VeladjInit;
                Flag_DisadjInit = tmpLocFlag_DisadjInit;
                Flag_TermInit = tmpLocFlag_TermInit;
                Flag_StopCond = tmpLocFlag_StopCond;

                ModelParam.accdif_disadj = tmpLoc_ModelParam_accdiff;
                ModelParam.dis_stall = tmpLoc_ModelParam_dis_stall;
                ModelParam.vel_stall = tmpLoc_ModelParam_vel_stall;
                ModelParam.velr_stall = tmpLoc_ModelParam_velr_stall;
                ModelParam.disr_term = tmpLoc_ModelParam_disr_term;                
                ModelParam.AccDiffIndex = tmpLocParam_AccDiffIndex;   
                ModProf_state(Index_timestep+1) = stState;
            end
            % tmpdis_debug_point = FcnDeb(index_curstep,141)
            tmpStateArry(index_curstep) = stState;
            tmpLocFlag_PreState = stState;                
       end % End for prediction iteration
       
       % Prediction results for profiles
       ModProf_dis_adj(Index_timestep+1:Index_ModLength) = tmpLoc_ModProf_dis_adj(Index_timestep+1:Index_ModLength);
       ModProf_dis_del(Index_timestep+1:Index_ModLength) = tmpLoc_ModProf_dis_del(Index_timestep+1:Index_ModLength);
       ModProf_acc_ref(Index_timestep+1:Index_ModLength) = tmpLoc_ModProf_acc_ref(Index_timestep+1:Index_ModLength);
       ModProf_vel_ref(Index_timestep+1:Index_ModLength) = tmpLoc_ModProf_vel_ref(Index_timestep+1:Index_ModLength);

       ModProf_acc_est(Index_timestep+1:Index_ModLength) =  tmpLoc_Mod_acc_est(Index_timestep+1:Index_ModLength);
       ModProf_vel_est(Index_timestep+1:Index_ModLength) =  tmpLoc_Mod_vel_est(Index_timestep+1:Index_ModLength);
       ModProf_dis_est(Index_timestep+1:Index_ModLength) =  tmpLoc_Mod_dis_est(Index_timestep+1:Index_ModLength);                      

      
       % Data storage for prediction
       % --- Braking algorithm uses only 1st step
       if  Index_timestep == PreIndex
           indexlen_prestep = (ParamSel.Tp_Term + OnlineConfig.TermBuff) - Index_timestep;                                                
           ArryModProf_dis_adj(Index_timestep:Index_ModLength,k) = tmpLoc_ModProf_dis_adj(Index_timestep:Index_ModLength);
           ArryModProf_dis_del(Index_timestep:Index_ModLength,k) = tmpLoc_ModProf_dis_del(Index_timestep:Index_ModLength);
           ArryModProf_acc_ref(Index_timestep:Index_ModLength,k) = tmpLoc_ModProf_acc_ref(Index_timestep:Index_ModLength);
           ArryModProf_vel_ref(Index_timestep:Index_ModLength,k) = tmpLoc_ModProf_vel_ref(Index_timestep:Index_ModLength);


           ArryModProf_acc_est(Index_timestep:Index_ModLength,k) =  tmpLoc_Mod_acc_est(Index_timestep:Index_ModLength);
           ArryModProf_vel_est(Index_timestep:Index_ModLength,k) =  tmpLoc_Mod_vel_est(Index_timestep:Index_ModLength);
           ArryModProf_dis_est(Index_timestep:Index_ModLength,k) =  tmpLoc_Mod_dis_est(Index_timestep:Index_ModLength);
           ArryModProf_acc_veff(Index_timestep:Index_ModLength,k) =  tmpLoc_Mod_acc_est_veff(Index_timestep:Index_ModLength);
           ArryModProf_acc_deff(Index_timestep:Index_ModLength,k) =  tmpLoc_Mod_acc_est_deff(Index_timestep:Index_ModLength);               

           ArryModProf_State(Index_timestep:Index_ModLength-1,k) =  tmpStateArry(Index_timestep:Index_ModLength-1);
           k = k+1; if k>=length(PreIndexSet) k=length(PreIndexSet); end; PreIndex = PreIndexSet(k);
       end        
    end % End for time iteration        
    % Estimation result store 
    % ----- Save whole prediction at the 1st step
    McEstResult(:,1) = ArryModProf_acc_est;
    McEstResult(:,2) = ArryModProf_acc_ref;    
    McEstResult(:,3) = ArryModProf_vel_est;
    McEstResult(:,4) = ArryModProf_vel_ref;
    McEstResult(:,5) = ArryModProf_dis_est;
    McEstResult(:,6) = ArryModProf_dis_adj;
    McEstResult(:,7) = ArryModProf_dis_del;
    McEstResult(2:end,8) = ArryModProf_State;
    % ----- Save each prediction for the each step    
    TdEstResult(:,1) = ModProf_acc_est;
    TdEstResult(:,2) = ModProf_acc_ref;
    TdEstResult(:,3) = ModProf_vel_est;
    TdEstResult(:,4) = ModProf_vel_ref;
    TdEstResult(:,5) = ModProf_dis_est;
    TdEstResult(:,6) = ModProf_dis_adj;
    TdEstResult(:,7) = ModProf_dis_del;
    TdEstResult(:,8) = ModProf_state;    
    eval(['EstResult.Case' num2str(Indexcase) '_Mc = McEstResult;'])
    eval(['EstResult.Case' num2str(Indexcase) '_Td = TdEstResult;'])
    %% Learning algorithm for Parameter update
    % Determine time points 
    tmpTpAdj = find(ArryModProf_State == ModelConfig.stState.Disadj,1);    
    tmpTpTerm = find(ArryModProf_State == ModelConfig.stState.Term,1);
    
    tmpVelEff = (1 - VehData_acc./ParamSel.MaxAcc);
    tmpVelEff(tmpVelEff<LrnConfig.VelEffMin) = LrnConfig.VelEffMin;    
    tmpTpStop = find((VehData_vel<LrnConfig.VelTerm) & (VehData_time>LrnConfig.CoastTime),1);
    if isempty(tmpTpStop)
        tmpTpStop = ParamSel.Tp_Term;
    end
    if isempty(tmpTpTerm)
        tmpTpTerm = ParamSel.Tp_Term;
    end
    tmpTpBrk = find(VehData_brk>2,1);
    
    % Calculate reference profile using braking data
    LrnProfVelRef = VehData_vel./(tmpVelEff).^(1/ModelConfig.Delta);        
    LrnProfVelDiff = VehData_vel - LrnProfVelRef;
    LrnProfVelDiffFilt = sgolayfilt(LrnProfVelDiff,1,LrnConfig.AccFiltNum);
    LrnProfAccFilt = sgolayfilt(VehData_acc,1,ModelConfig.AccFiltNum);
    % For Acc Ref,
    %   --- Using the calculated data from braking acceleration = tmpData(:,18)
    LrnProfAccRef = -0.5*VehData_vel.^2./(VehData_dis - ModelConfig.StopDis);    
    LrnProfAccRefFilt = sgolayfilt(LrnProfAccRef,1,ModelConfig.AccFiltNum);
    LrnProfAccRefFiltComp = tmpData(:,18);
    LrnProfAccRefFiltComp(end:Index_ModLength) = LrnProfAccRefFiltComp(end);   
    
    % Determine time point using reference profiles
    %   --- Initial point = Braking point
    %   --- Adjustment point = Max(VelDiff)
    tmpTpInit = find(LrnProfVelDiffFilt(1:tmpTpStop)>=LrnProfVelDiffFilt(1)*LrnConfig.VelDifRatFac_Init,1);                              
    tmpVpDiffInit = LrnProfVelDiffFilt(tmpTpInit);
    [tmpDummy tmpTpMax] = max(LrnProfVelDiffFilt(tmpTpBrk:tmpTpTerm));
    tmpTpDelta = find(LrnProfVelDiffFilt(tmpTpBrk:tmpTpTerm)>=tmpVpDiffInit+LrnConfig.VelDeltaOff,1);        
    if isempty(tmpTpDelta)
        tmpTpDelta = tmpTpMax;
    end         
    tmpTpDelta = tmpTpMax-1;        
    tmpTpStall = tmpTpMax + tmpTpBrk - 1;
    
    % Determine acceleration points at adjustment start
    %   --- AccDiff = AccRef(TpInit) - AccRef(TpAdj)
    %   --- AccRatio = AccEst(TpAdj)/AccRef(TpAdj)
    tmpAppStall = mean(LrnProfAccFilt(tmpTpStall-LrnConfig.AcrCalcWindow:tmpTpStall+LrnConfig.AcrCalcWindow));
    tmpAppRefInit = LrnProfAccRefFiltComp(tmpTpBrk);
    tmpAppRefStall = LrnProfAccRefFiltComp(tmpTpStall);
    
    % Determine termination points
    %   --- Time point when AccRef converges to AccRef (when AccRatio < 1)
    %   --- Stop point - 30(magic number!!) (when AccRatio > 1)
    tmpAccCompPtSt = min(tmpTpAdj,tmpTpStall) + LrnConfig.TermBuff;
    tmpTpAdjRefIndex = find(abs(LrnProfAccRefFiltComp(tmpAccCompPtSt:tmpTpStop) - LrnProfAccFilt(tmpAccCompPtSt:tmpTpStop))<=LrnConfig.AdjAccDiff , 1);
    if isempty(tmpTpAdjRefIndex)
        tmpTpAdjRefIndex = tmpTpStop;
    end
    tmpTpAdjRef = min(tmpTpAdjRefIndex);
    tmpTpTermRef = tmpAccCompPtSt+tmpTpAdjRef-1;
    if (tmpAppStall/tmpAppRefStall >= 0.99)
        tmpTpTermRef = tmpTpStop - 30;
    end
    
    % Assign learning parameters
    Lrn_ExpParam_TpInit = tmpTpBrk;
    Lrn_ExpParam_TpStall = tmpTpStall;
    Lrn_ExpParam_TpDelta = tmpTpDelta;
    Lrn_ExpParam_VpStall = LrnProfVelRef(tmpTpStall);
    Lrn_ExpParam_AprStall = tmpAppStall/tmpAppRefStall;    
    Lrn_ExpParam_ApdStall = tmpAppRefStall - tmpAppRefInit;
    Lrn_ExpParam_MaxAcc = -min(LrnProfAccFilt);
    Lrn_ExpParam_TpTerm = tmpTpTerm;
    Lrn_ExpParam_TpTermRef = tmpTpTermRef;
    
    % Storage parameters
    ModelParam.LrnParamArry(Indexcase,1) = Lrn_ExpParam_MaxAcc;
    ModelParam.LrnParamArry(Indexcase,2) = Lrn_ExpParam_AprStall;
    ModelParam.LrnParamArry(Indexcase,3) = Lrn_ExpParam_TpInit;
    ModelParam.LrnParamArry(Indexcase,4) = Lrn_ExpParam_TpDelta+Lrn_ExpParam_TpInit;
    ModelParam.LrnParamArry(Indexcase,5) = Lrn_ExpParam_ApdStall;
    ModelParam.LrnParamArry(Indexcase,6) = Lrn_ExpParam_TpTermRef;
    ModelParam.LrnParamArry(Indexcase,7) = Lrn_ExpParam_TpTerm;
    ModelParam.Lrn_TpStop(Indexcase) = tmpTpStop;
    
    
    % Update parameter vector
    %   --- Calculate parameter update value using learning degree
    LrnParamVecDelta_AccMax = Fcn_FzUpdate(Lrn_ExpParam_MaxAcc,ModelParam.MaxAcc,ModelParam.IndexProbVec_Vel,LrnConfig.ExpUpdateGain)';
    LrnParamVecDelta_AprStall = Fcn_FzUpdate(Lrn_ExpParam_AprStall,ModelParam.AprStall,ModelParam.IndexProbVec_Acc,LrnConfig.ExpUpdateGain)';
    LrnParamVecDelta_TpInit = Fcn_FzUpdate(Lrn_ExpParam_TpInit,ModelParam.TpInit,ModelParam.IndexProbVec_Acc,LrnConfig.ExpUpdateGain)';
    LrnParamVecDelta_TpDelta = Fcn_FzUpdate(Lrn_ExpParam_TpDelta,ModelParam.TpDelta,ModelParam.IndexProbVec_Acc,LrnConfig.ExpUpdateGain)';
    LrnParamVecDelta_ApdStall = Fcn_FzUpdate(Lrn_ExpParam_ApdStall,ModelParam.ApdStall,ModelParam.IndexProbVec_Acc,LrnConfig.ExpUpdateGain)';
    %   --- Update parameter vector
    LrnParamVec_AccMax = LrnParamVec_AccMax + LrnParamVecDelta_AccMax;
    LrnParamVec_AprStall = LrnParamVec_AprStall + LrnParamVecDelta_AprStall;
    LrnParamVec_TpInit = LrnParamVec_TpInit + LrnParamVecDelta_TpInit;
    LrnParamVec_TpDelta = LrnParamVec_TpDelta + LrnParamVecDelta_TpDelta;
    LrnParamVec_ApdStall = LrnParamVec_ApdStall + LrnParamVecDelta_ApdStall;
    % Data storage
    ModelParam.LrnArry_MinAccVec(:,Indexcase) = LrnParamVec_AccMax;
    ModelParam.LrnArry_AprStall(:,Indexcase) = LrnParamVec_AprStall;
    ModelParam.LrnArry_ApdStall(:,Indexcase) = LrnParamVec_ApdStall;
    ModelParam.LrnArry_TpInit(:,Indexcase) = LrnParamVec_TpInit;
    % Adjust parameter update - reference acceleration tracking gain    
    LrnParamVecDelta_AdjGain = Fcn_FzUpdate(Lrn_ExpParam_TpTerm,Lrn_ExpParam_TpTermRef,ModelParam.IndexProbVec_Vel,LrnConfig.AdjUpdateGain)';
    LrnParamVec_AdjGain = LrnParamVecDelta_AdjGain + LrnParamVec_AdjGain;
    LrnParamVec_AdjGain(LrnParamVec_AdjGain<=LrnConfig.AdjLimLow) = LrnConfig.AdjLimLow;
    
    ModelParam.LrnArry_AdjGain(:,Indexcase) = LrnParamVec_AdjGain;
    Arry_acc_est(:,Indexcase) = ArryModProf_acc_est;
    Arry_vel_ref(:,Indexcase) = ArryModProf_vel_ref;
    Arry_vel_est(:,Indexcase) = ArryModProf_vel_est;
    clearvars tmp*
end % End for case iteration
%% Result plot
figure;plot(ArryModProf_acc_est)
hold on
plot(VehData_acc,'--','linewidth',2)
plot(Arry_acc_est(:,1),'--','linewidth',2)
plot(LrnProfAccRefFilt,'--','linewidth',2)
plot(LrnProfAccFilt,'-.','linewidth',2)
plot(LrnProfAccRefFiltComp,'-.','linewidth',4)
plot(Lrn_ExpParam_TpTermRef,ArryModProf_acc_ref(Lrn_ExpParam_TpTermRef),'o')
plot(ArryModProf_acc_ref,'--')
%%
