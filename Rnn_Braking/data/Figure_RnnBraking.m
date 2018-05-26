% Journal figure plot
% --- Title: RNN model for braking prediction
% --- 2018/05/14
clear all; close all;close all;
load('PreResult.mat')
KH_ColorCode
%% Fig. 1.Braking scenario
%% Fig. 2 Braking data
tmpIndexSet = [1 10 40];
close all
l = 1;
for j = 1:3
    fig = figure('name','','numbertitle','off');
    set(gcf,'Color',[1,1,1],'position',[500 200 300 250])
    tmpIndex = tmpIndexSet(j);
    tmpData = Driver_1.Data{1,tmpIndex};
    tmpTime = tmpData(:,10);
    tmpAcc =  tmpData(:,6);
    tmpVeh = tmpData(:,7);
    tmpDis = tmpData(:,8);
    xlimset = [0 max(tmpTime)];

    plot(tmpTime,tmpAcc,'linewidth',1.5);grid on;
    xlabel('Time [s]')
    ylabel('Acceleration [m/s^2]');
%     ylim([-5 1]);
    xlim(xlimset)
end
%% Fig. 3. Sequential model
%% Fig. 4. Model structure
%% Fig. 5. AccRef data
tmpIndexSet = [1 10 40];
close all
for j = 1:3
    fig = figure('name','','numbertitle','off');
    set(gcf,'Color',[1,1,1],'position',[500 200 300 250])
    tmpIndex = tmpIndexSet(j);
    tmpData = Driver_1.Data{1,tmpIndex};
    tmpTime = tmpData(:,10);
    tmpAcc =  tmpData(:,6);
    tmpAccRef = tmpData(:,9);    
    xlimset = [0 max(tmpTime)];
    plot(tmpTime,tmpAcc,'linewidth',1.5);grid on;hold on;
    plot(tmpTime,tmpAccRef,'linewidth',1.5);grid on;hold on;
    xlabel('Time [s]')
    ylabel('Acceleration [m/s^2]');
%     ylim([-5 1]);
    xlim(xlimset)
end
legend('Acceleration','Reference acceleration')
%% Fig. 6. TPE
%% Fig. 7. Param opt result
load('ParamOptResult.mat')
for i = 1:250
    eval(['tmpIterationData = Iteration_' num2str(i-1) ';'])
    score(i) = tmpIterationData.val_loss(end);
    if i == 1
        min_score = score(i);
    else
        min_score(i) = min(min_score(i-1),score(i));
    end
end

fig = figure('name','','numbertitle','off');
set(gcf,'Color',[1,1,1],'position',[500 200 500 400])
hold on;grid on;
mk = scatter(1:250,score,'o','MarkerFaceColor',Color.SP(7,:),'MarkerFaceAlpha',0.5,'SizeData',15);hold on;
plot(1:250,min_score,'linewidth',1.5,'color',Color.SP(10,:))
ylim([0 0.01])
xlabel('Iteration [-]')
ylabel('Score = MSE')
legend('TPE score','Min score value')

clear vars Iter* tmp*
%% Fig. 8. Prediction process
%% Fig. 9. Prediction results
tmpIndexSet = [1 10 40];
close all
for j = 1:3
    fig(j) = figure('name','','numbertitle','off');
    set(gcf,'Color',[1,1,1],'position',[500 200 300 250])    
    tmpIndex = tmpIndexSet(j);    tmpData = Driver_1.Data{1,tmpIndex};
    tmpTime = tmpData(:,10);
    tmpAcc =  tmpData(:,6);
    tmpAccPredic =  tmpData(:,1);
    tmpAccRefPredic = tmpData(:,4);    
    xlimset = [0 max(tmpTime)];
    plot(tmpTime,tmpAcc,'linewidth',1.5);grid on;hold on;
    plot(tmpTime(13:end),tmpAccPredic(13:end),'linewidth',1.5);grid on;hold on;
    plot(tmpTime(13:end),tmpAccRefPredic(13:end),'linewidth',1.5);grid on;hold on;
    xlabel('Time [s]')
    ylabel('Acceleration [m/s^2]');
    xlim(xlimset)
end
legend('Measurement','Prediction','Calculated reference')


for j = 1:3
    fig(j) = figure('name','','numbertitle','off');
    set(gcf,'Color',[1,1,1],'position',[500 200 300 250])    
    tmpIndex = tmpIndexSet(j);    tmpData = Driver_1.Data{1,tmpIndex};
    tmpTime = tmpData(:,10);
    tmpVel =  tmpData(:,7);
    tmpVelPredic =  tmpData(:,2);    
    xlimset = [0 max(tmpTime)+1];
    plot(tmpTime,tmpVel,'linewidth',1.5);grid on;hold on;
    plot(tmpTime(13:end),tmpVelPredic(13:end),'linewidth',1.5);grid on;hold on;    
    xlabel('Time [s]')
    ylabel('Velocity [m/s]');
    xlim(xlimset)
end
legend('Measurement','Prediction')
%% Parameter arrangement
ParamLst_MaxAcc = [];
ParamLst_VelCst = [];
ParamLst_AccDiff = [];
ParamLst_MaxPnt = [];
ParamLst_Cluster = [];

for j = 1:3
    eval(['tmpDriverData = Driver_' num2str(j) ';'])
    ParamLst_AccDiff = [ParamLst_AccDiff tmpDriverData.Par_AccDif];
    ParamLst_MaxPnt = [ParamLst_MaxPnt tmpDriverData.Par_MaxPnt];
    ParamLst_Cluster = [ParamLst_Cluster tmpDriverData.CluIndex];
    tmpCaseLen = length(tmpDriverData.DataIndex);
    tmpMaxAcc = [];tmpVelCst = [];
    for i = 1:tmpCaseLen
        tmpData = tmpDriverData.Data{1,i};
        tmpAcc = tmpData(:,6);
        tmpVel = tmpData(:,7);

        tmpMaxAcc(i) = min(tmpData(:,6));
        tmpVelCst(i) = tmpData(1,7);
    end
    ParamLst_MaxAcc = [ParamLst_MaxAcc tmpMaxAcc];
    ParamLst_VelCst = [ParamLst_VelCst tmpVelCst];
    eval(['Driver_' num2str(j) '.Par_MaxAcc = tmpMaxAcc;'])
    eval(['Driver_' num2str(j) '.Par_VelCst = tmpVelCst;'])
end
ParamLst_ParCst = ParamLst_MaxAcc./ParamLst_VelCst;

DriverIndex(1:103) = 1;
DriverIndex(104:203) = 2;
DriverIndex(204:297) = 3;
%% Fig. 10. Param Max Acc vs V coast
DriverColor = [Color.BP(3,:);Color.GP(10,:);Color.BP(10,:)];
MarkerSet = ['o','^','<'];
fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 400 350]);
for j = 1:3
    s = scatter(ParamLst_VelCst(DriverIndex == j),ParamLst_MaxAcc(DriverIndex == j),MarkerSet(j),'MarkerEdgeColor',DriverColor(j,:),'MarkerFaceColor',DriverColor(j,:),'MarkerFaceAlpha',0.5,'SizeData',25);hold on;grid on;
end
legend('Driver 1','Driver 2','Driver 3')
xlabel('Coasting velocity [m/s]')
ylabel('Maximum decceleration [m/s^2]')

fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 400 350]);
for j = 1:3
    s = scatter(ParamLst_VelCst(DriverIndex == j),ParamLst_MaxAcc(DriverIndex == j)./ParamLst_VelCst(DriverIndex == j),MarkerSet(j),'MarkerEdgeColor',DriverColor(j,:),'MarkerFaceColor',DriverColor(j,:),'MarkerFaceAlpha',0.5,'SizeData',25);hold on;grid on;
end
legend('Driver 1','Driver 2','Driver 3')
xlabel('Coasting velocity [m/s]')
ylabel('Coasting Param [1/s]')
%% Fig. 11. Ref velocity profile
close all
tmpVelRef = FcnRefVelCalc(tmpAcc,tmpVel);
tmpVelDiffFilt = tmpVel - tmpVelRef;

for j = 1:3
    fig(j) = figure('name','','numbertitle','off');
    set(gcf,'Color',[1,1,1],'position',[500 200 300 550])    
    tmpIndex = tmpIndexSet(j);
    tmpData = Driver_1.Data{1,tmpIndex};
    tmpTime = tmpData(:,10);
    tmpAcc =  tmpData(:,6);
    tmpVel =  tmpData(:,7);
    tmpAccRef = tmpData(:,9);
    tmpVelRef = FcnRefVelCalc(tmpAcc,tmpVel);
    tmpVelDiffFilt = tmpVel - tmpVelRef;
    
    [tmpDiffVelMax tmpMaxPnt] = max(tmpVelDiffFilt);
    xlimset = [0 max(tmpTime)];
    subplot(2,1,1)
    plot(tmpTime,tmpAcc,'linewidth',1.5);grid on;hold on;
    plot(tmpTime,tmpAccRef,'linewidth',1.5);grid on;hold on;
    plot(tmpTime(tmpMaxPnt),tmpAcc(tmpMaxPnt),'o','markerfacecolor',Color.BP(3,:),'color',Color.BP(2,:));grid on;hold on;    
    ylabel('Acceleration [m/s^2]');
    legend('Measurement','Reference')
    xlim(xlimset)
    subplot(2,1,2)
    plot(tmpTime,tmpVelDiffFilt,'linewidth',1.5);grid on;hold on;    
    h = plot(tmpTime(tmpMaxPnt),tmpVelDiffFilt(tmpMaxPnt),'o','markerfacecolor',Color.BP(3,:),'color',Color.BP(2,:));grid on;hold on;    
    legend(h, 'Maximum Point')
    xlabel('Time [s]')
    ylabel('Velocity Profile [m/s]');
    xlim(xlimset)
end
%% Fig. 12. K-means clustering
% Parameter X = Max ponint
% Parameter Y = Acc Diff
% Parameter Z = Coast param

ClColorSet = [Color.SP(2,:);Color.CP(6,:);Color.GP(4,:);Color.BP(5,:)];
fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 400 350]);
for i = 1:4
    tmpIndexClu = ParamLst_Cluster == i-1;
    scatter(ParamLst_MaxPnt(tmpIndexClu),ParamLst_AccDiff(tmpIndexClu),'MarkerEdgeColor',ClColorSet(i,:),'MarkerFaceColor',ClColorSet(i,:),'MarkerFaceAlpha',0.5,'SizeData',25);hold on;grid on;
end

fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 400 350]);
for i = 1:4
    tmpIndexClu = ParamLst_Cluster == i-1;
    scatter(ParamLst_MaxPnt(tmpIndexClu),ParamLst_ParCst(tmpIndexClu),'MarkerEdgeColor',ClColorSet(i,:),'MarkerFaceColor',ClColorSet(i,:),'MarkerFaceAlpha',0.5,'SizeData',25);hold on;grid on;
end

fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 400 350]);
for i = 1:4
    tmpIndexClu = ParamLst_Cluster == i-1;
    scatter(ParamLst_ParCst(tmpIndexClu),ParamLst_AccDiff(tmpIndexClu),'MarkerEdgeColor',ClColorSet(i,:),'MarkerFaceColor',ClColorSet(i,:),'MarkerFaceAlpha',0.5,'SizeData',25);hold on;grid on;
end


fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 400 350]);
for i = 1:4
    tmpIndexClu = ParamLst_Cluster == i-1;
    scatter3(ParamLst_MaxPnt(tmpIndexClu),ParamLst_AccDiff(tmpIndexClu),ParamLst_ParCst(tmpIndexClu),'MarkerEdgeColor',ClColorSet(i,:),'MarkerFaceColor',ClColorSet(i,:),'MarkerFaceAlpha',0.5,'SizeData',25);hold on;grid on;
end
xlabel('Max point [s]')
ylabel('Acc Diff [m/s^2]')
ylabel('Acc Diff [m/s^2]')
zlabel('Coasting Param [1/s]')
%% Fig. 14. Braking according to clustering
%% Fig. 15. Driver cluster
% FaceColor = [Color.CP(7,:);Color.GP(5,:);Color.BP(8,:)];
% EdgeColor = [Color.CP(6,:);Color.SP(6,:);Color.BP(5,:)];
FaceColor = DriverColor;
EdgeColor = DriverColor;
fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 600 700 300]);
for i = 1:3
    axes(i) = subplot(1,3,i);hold on;grid on;
end
   

% fig2 = figure();
% ax4 = axes(fig2);hold on;grid on;
for k = 1:3
    eval(['tmpDataStr = Driver_' num2str(k) ';'])   
    h(k) = histogram(axes(k),tmpDataStr.CluIndex,'FaceAlpha',0.2,'FaceColor',Color.SP(k*2,:));hold on;
    bar_val(:,k) = h(k).Values;
end    

close all

fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 400 350]);
b = bar(1:4,bar_val);grid on;
b(1).LineWidth = 1.5; b(1).FaceColor = FaceColor(1,:); b(1).FaceAlpha = 0.5; b(1).LineStyle = '-'; b(1).EdgeColor = EdgeColor(1,:);
b(2).LineWidth = 1.5; b(2).FaceColor = FaceColor(2,:); b(2).FaceAlpha = 0.5; b(2).LineStyle = ':'; b(2).EdgeColor = EdgeColor(2,:);
b(3).LineWidth = 1.5; b(3).FaceColor = FaceColor(3,:); b(3).FaceAlpha = 0.5; b(3).LineStyle = '-.'; b(3).EdgeColor = EdgeColor(3,:);
legend('Driver1', 'Driver2', 'Driver3')
xlabel('Cluster index [-]')
ylabel('Case number [-]')
%%

fig3 = figure();
set(fig3,'Color',[1,1,1],'Position',[2000 100 600 400])
for i = 1:4
    ax(i) = subplot(2,2,i);hold on;grid on;
end

for k = 1:3
    eval(['tmpDataStr = Driver_' num2str(k) '']);
    for i=1:4
        data_index = SelectIndex(i,k);
        tmp_Data = tmpDataStr.Data{1,data_index};
        plot(ax(i),tmp_Data(:,10),tmp_Data(:,6),'--','Display','Acc Measurement','Color',DriverColor(k,:));        
        plot(ax(i),tmp_Data(:,10),tmp_Data(:,9),':','Display','AccRef Measurement','Color',DriverColor(k,:));
        plot(ax(i),tmp_Data(tmpDataStr.Par_MaxPnt(data_index),10),tmp_Data(tmpDataStr.Par_MaxPnt(data_index),6),'o','Color',DriverColor(k,:));
    end
end
%%

%%
function VelRef = FcnRefVelCalc(Acc,Vel)
    MaxAcc = -min(Acc);
    tmpAccDem = (1 - Acc./MaxAcc);
    tmpAccDem(tmpAccDem<=0) = 0.00001;
    VelRef = Vel./(tmpAccDem.^0.25);
end    
    
