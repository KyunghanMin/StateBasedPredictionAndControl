function [BlendedParam,ProbVecOut] = Fcn_FzBlending(Data,DataIndex,ParamSet)
%FCN_BLENDING Summary of this function goes here
%   Detailed explanation goes here

IndexLength = length(DataIndex);
LrnConfig.LrnStd = 3;

% CoastSpeed = 0:0.01:100;

for i=1:IndexLength
    ProbVec(:,i) = 1/(LrnConfig.LrnStd*sqrt(2*pi))*exp(-0.5*((Data - DataIndex(i))./LrnConfig.LrnStd).^2);
end

ProbSum = sum(ProbVec);
ProbNorm = ProbVec./ProbSum;

BlendedParam = ProbNorm*ParamSet;

ProbVecOut = ProbNorm;

end

