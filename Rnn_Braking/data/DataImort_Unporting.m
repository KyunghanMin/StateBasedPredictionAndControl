clear all
DriverSet = {'KH','YK','GB'};
ModelConfig_Sequence = 10;
k = 1;
DataLength = 0;
for j=1:3
    filename = ['Data_' char(DriverSet(j)) '.mat'];
    load(filename)
    
    for i=1:DataCoast.CoastNum
        eval(['tmpData = DataCoast.Case' num2str(i) ';']);
        tmpData_Scale = tmpData(1:10:end,:);
        tmpDataLen = length(tmpData_Scale);
        [tmpDummy tmpStopPoint] = min(tmpData_Scale(:,4));    
        tmpData_Form = tmpData_Scale(1:tmpStopPoint,:);
        endpoint = tmpStopPoint+ModelConfig_Sequence+1-rem(tmpStopPoint,ModelConfig_Sequence);
        tmpData_Acc = tmpData_Form(:,1);
        tmpData_Vel = tmpData_Form(:,4);
        tmpData_Dis = tmpData_Form(:,17);
        tmpData_Acc(end:endpoint) = tmpData_Acc(end);
        tmpData_Vel(end:endpoint) = tmpData_Vel(end);
        tmpData_Dis(end:endpoint) = tmpData_Dis(end);    
        tmpDataReArry = [tmpData_Acc(1:end-1)  tmpData_Vel(1:end-1)  tmpData_Dis(1:end-1)  tmpData_Acc(2:end)];
        tmpDataSetLeng = endpoint-ModelConfig_Sequence;
        DataLength = DataLength + tmpDataSetLeng-1;
        DataLengthArry(k) = tmpDataSetLeng;
        eval(['CaseData' num2str(k) ' = tmpDataReArry;'])
        k = k+1;
    end
end
save('TestUpData.mat','CaseData*','DataLength');