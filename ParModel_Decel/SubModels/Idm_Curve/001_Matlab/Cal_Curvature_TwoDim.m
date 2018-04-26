function TwoDim_CurvatureRoadModel=Cal_Curvature_TwoDim(B_SplineRoadModel,Location)
%% Curvature Parametric 


dB_SplineRoadModel = fnder(B_SplineRoadModel);
dVal_B_SplineRoadModel = single(fnval(dB_SplineRoadModel,Location));
ddB_SplineRoadModel = fnder(dB_SplineRoadModel);
ddVal_B_SplineRoadModel = single(fnval(ddB_SplineRoadModel,Location));
crossD_DD = cross( dVal_B_SplineRoadModel',ddVal_B_SplineRoadModel');


TwoDim_dVal_B_SplineRoadModel=dVal_B_SplineRoadModel;
TwoDim_ddVal_B_SplineRoadModel=ddVal_B_SplineRoadModel;
TwoDim_dVal_B_SplineRoadModel(3,:)=0;
TwoDim_ddVal_B_SplineRoadModel(3,:) = 0;
% CrossProductEle1=TwoDim_dVal_B_SplineRoadModel(:,1).*TwoDim_ddVal_B_SplineRoadModel(:,2);
% CrossProductEle2=TwoDim_dVal_B_SplineRoadModel(:,2).*TwoDim_ddVal_B_SplineRoadModel(:,1);

TwoDim_crossD_DD = cross(TwoDim_dVal_B_SplineRoadModel',TwoDim_ddVal_B_SplineRoadModel');
% TwoDim_crossD_DD = CrossProductEle1-CrossProductEle2;

%%


numCur = sqrt([1 1 1]*abs(crossD_DD.*crossD_DD)'); % Numerator of curvature
denCur = sqrt([1 1 1]*abs(dVal_B_SplineRoadModel.*dVal_B_SplineRoadModel.*dVal_B_SplineRoadModel)); % Denominator of curvature
ThreeDim_CurvatureRoadModel = numCur ./ denCur;



TwoDim_numCur = sqrt([1 1 1]*abs(TwoDim_crossD_DD.*TwoDim_crossD_DD)'); % Numerator of curvature
TwoDim_denCur = sqrt([1 1 1]*abs(TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel.*TwoDim_dVal_B_SplineRoadModel)); % Denominator of curvature
TwoDim_CurvatureRoadModel = TwoDim_numCur ./ TwoDim_denCur;