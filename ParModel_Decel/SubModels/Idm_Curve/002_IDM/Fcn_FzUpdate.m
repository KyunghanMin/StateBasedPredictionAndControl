function [LrnParamVec] = Fcn_FzUpdate(ExplicitParam,BlendedParam,ProbVec,UpdateGain)
%FCN_FZUPDATE Summary of this function goes here
%   Detailed explanation goes here

IndexLength = length(ProbVec);
IndexSet = 1:IndexLength;

TargetDelta = UpdateGain*(ExplicitParam - BlendedParam);

for i=1:IndexLength
    ProbSumEx(i) = sum( ProbVec(IndexSet ~= i) );    
end

TmpParamSum = sum(ProbVec./ProbSumEx);
% Calculate learning degree
for i=1:IndexLength
    LrnParamFac(i) = ProbSumEx(i)*TmpParamSum;
end    

LrnParamVec = TargetDelta./LrnParamFac;

end

