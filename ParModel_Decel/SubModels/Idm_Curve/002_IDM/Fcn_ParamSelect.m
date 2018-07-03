function [ParamCase] = Fcn_ParamSelect(ParamSet,Index)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

% Tmp index searching

ParamCase.Aprat_Stall  = ParamSet.Aprat_Stall(Index);
ParamCase.Ap_Min  = ParamSet.Ap_Min(Index);

ParamCase.Tp_Shift = ParamSet.Tp_Shift(Index);
ParamCase.Tp_Init = ParamSet.Tp_Init(Index);
ParamCase.Tp_Stall = ParamSet.Tp_Stall(Index);
ParamCase.Tp_Term = ParamSet.Tp_Term(Index);

ParamCase.Vp_Coast = ParamSet.Vp_Coast(Index);
ParamCase.Vp_Stall = ParamSet.Vp_Stall(Index);

ParamCase.Vp_Delta = ParamSet.Vp_Delta(Index);
ParamCase.Tp_Delta = ParamSet.Tp_Delta(Index);
ParamCase.Delta = ParamSet.Delta(Index);

ParamCase.ApFact = ParamSet.ApFact(Index);
ParamCase.AccDelta = ParamSet.ApDelta(Index);
ParamCase.AccDiff = ParamSet.Apdiff_Stall(Index);

end

