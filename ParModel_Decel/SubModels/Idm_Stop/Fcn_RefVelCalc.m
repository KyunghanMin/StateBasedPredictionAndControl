function [Mod_RefVel,Mod_DisEff,Mod_VelEff] = Fcn_RefVelCalc(ModelConfig, LrnConfig, Data, AccMin)
%FCN_REFVELCALC Summary of this function goes here
%   Detailed explanation goes here


veh_a = Data(:,1);
veh_v = Data(:,2);
veh_dis = Data(:,3);


par_a_max = abs(AccMin);
par_dleta = ModelConfig.Delta;
par_d0 = ModelConfig.MinDis;
par_T = ModelConfig.TimeGap;

Mod_DisEff = ((par_d0 + par_T.*veh_v)./veh_dis).^2;
Mod_VelEff = (1 - veh_a./par_a_max);

Mod_VelEff(find((Mod_VelEff<=LrnConfig.VelEffMin))) = LrnConfig.VelEffMin;
Mod_RefVel = veh_v./(Mod_VelEff.^(1/par_dleta));

end

