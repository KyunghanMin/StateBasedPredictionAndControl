clear all
DriverSet = {'KH','YK','GB'};
ModelConfig_Sequence = 10;
k = 1;
DataLength = 0;
xx_flag = 0;
for j=1:3
    filename = ['Data_' char(DriverSet(j)) '_Rnn.mat'];
    load(filename)
    
    for i=1:DataCoast.CoastNum
        eval(['tmpData = DataCoast.Case' num2str(i) ';']);
        tmpData_Scale = tmpData(1:10:end,:);
        tmpDataLen = length(tmpData_Scale);
        [tmpDummy tmpStopPoint] = min(tmpData_Scale(:,4));    
        tmpData_Form = tmpData_Scale(1:tmpStopPoint,:);
        tmpData_Acc = tmpData_Form(:,1);
        tmpData_Vel = tmpData_Form(:,4);
        tmpData_Dis = tmpData_Form(:,17);
%         tmpData_AccRef = tmpData_Form(:,18);
        tmpData_Time = tmpData_Form(:,19);
        tmpData_VelCalc = tmpData_Form(:,16);
        tmpData_DisCalc = tmpData_Form(:,17);
        tmpData_AccRef = -0.5*tmpData_VelCalc.^2./tmpData_DisCalc;        
        tmpDataReArry = [tmpData_Acc(1:end-1)  tmpData_Vel(1:end-1)  tmpData_Dis(1:end-1) tmpData_AccRef(1:end-1) tmpData_Time(1:end-1) tmpData_Acc(2:end)];        
        eval(['CaseData' num2str(k) ' = tmpDataReArry;'])
        k = k+1;
    end
end
save('TestUpDataVariRan.mat','CaseData*','DataLength');