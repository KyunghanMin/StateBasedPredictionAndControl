DriverSet = {'KH','YK','GB'};

i = 1;

DataIndex = 4;
DataArry(1,DataIndex) = 0;

for i=1:3
FileName = ['Data_' char(DriverSet(i)) '.mat'];
load(FileName);
    for j = 1:DataCoast.CoastNum
        eval(['tmpData = DataCoast.Case' num2str(j) ';'])
        tmpData_Scale = tmpData(1:10:end,:);
        [tmpDummy tmpStopPoint] = min(tmpData_Scale(:,4));
        IndexDataLen = tmpStopPoint;
        tmpData_Acc(1:IndexDataLen-1,1) = tmpData_Scale(1:IndexDataLen-1,1);
        tmpData_Vel(1:IndexDataLen-1,1) = tmpData_Scale(1:IndexDataLen-1,4);
        tmpData_Dis(1:IndexDataLen-1,1) = tmpData_Scale(1:IndexDataLen-1,17);
        tmpData_AccRef(1:IndexDataLen-1,1) = tmpData_Scale(2:IndexDataLen,1);
        DataArry = [DataArry; tmpData_Acc tmpData_Vel tmpData_Dis tmpData_AccRef];
    end
end