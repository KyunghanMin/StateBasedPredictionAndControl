function [stState] = Fcn_StateEstimation(index_curstep, ParamCase, tmpModProf, tmpLocFlag_StopState, tmpPreState)
%UNTITLED Summary of this function goes here
% Input 
% tmpModProf = [tmpLoc_Mod_acc_est(index_curstep) tmpLoc_ModProf_acc_ref(index_curstep) tmpLoc_ModProf_vel_est(index_curstep) tmpLoc_ModProf_vel_ref(index_curstep) tmpLoc_Mod_dis_est(index_curstep) tmpLoc_ModProf_dis_adj(index_curstep) tmpLoc_ModProf_dis_del(index_curstep)];                
% tmpLocFlag = tmpLocFlag_StopState;
%   Detailed explanation goes here
% Output
% Braking section
%      Coast: 1
%       Init: 2
%     Veladj: 3 - Adjustment section
%     Disadj: 4 - Adjustment section
%       Term: 5
%        End: 6


acc_est = tmpModProf(1);
acc_ref = tmpModProf(2);
dis_est = tmpModProf(5);
dis_adj = tmpModProf(6);

tmpLocFlag_StopCondi = tmpLocFlag_StopState;
tmpLocAccInitRef = acc_ref*ParamCase.Aprat_Stall;


if (index_curstep < ParamCase.Tp_Init) && (tmpPreState < 2)
    stState = 1; % Coast state
elseif (tmpLocAccInitRef <= acc_est) && (index_curstep <= (ParamCase.Tp_Init + ParamCase.Tp_Delta)) && (tmpPreState < 3)
    stState = 2; % Initial state
else
    stState = 3; % Stall - velocity adj
end

if tmpPreState == 5
    stState = 5;
elseif (dis_adj >= dis_est)
    if (tmpLocFlag_StopCondi == 1)&&(acc_est<=acc_ref)
        stState = 5;
    elseif (tmpLocFlag_StopCondi == -1)&&(acc_est>=acc_ref)
        stState = 5;
    else 
        stState = 4;    
    end 
end


end

