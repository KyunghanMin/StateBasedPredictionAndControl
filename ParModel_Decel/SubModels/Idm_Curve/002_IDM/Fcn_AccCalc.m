function [Mod_AccEst Mod_AccVelEffEst Mod_AccDisEffEst] = Fcn_AccCalc(tmpModProf,ModelConfig,ParamSel)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
Mod_VelEst = tmpModProf(3);
Mod_VelRef = tmpModProf(4);
Mod_DisEst = tmpModProf(5);
dis_adj = tmpModProf(6);
dis_delta = tmpModProf(7);

Mod_AccVelEffEst = (Mod_VelEst/Mod_VelRef).^ModelConfig.Delta;
Mod_AccDisEffEst = ((ModelConfig.MinDis+ModelConfig.AdjDis+dis_adj+ModelConfig.TimeGap*Mod_VelEst + dis_delta)/Mod_DisEst)^2;
Mod_AccEst = ParamSel.MaxAcc*(1 - Mod_AccVelEffEst - Mod_AccDisEffEst);
end

