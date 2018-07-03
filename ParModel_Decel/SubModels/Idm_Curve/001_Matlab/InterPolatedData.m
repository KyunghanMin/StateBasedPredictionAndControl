function [InterPolatedData] =InterPolatedData(Source, Target)
SourceLength = length(Source);
TargetLength = length(Target);
InterPolatedData = interp1( linspace(0,1,TargetLength), Target, linspace(0,1,SourceLength) );
