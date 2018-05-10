KH_ColorCode

load('PreResult')

fig = figure();
set(fig,'Color',[1,1,1],'Position',[500 300 700 300]);
ax1 = subplot(1,3,1);hold on;grid on;
ax2 = subplot(1,3,2);hold on;grid on;
ax3 = subplot(1,3,3);hold on;grid on;


fig2 = figure();
ax4 = axes(fig2);hold on;grid on;
for k = 1:3
    eval(['tmpDataStr = Driver_' num2str(k) ''])
    tmpX = tmpDataStr.Par_MaxPnt;
    tmpY = tmpDataStr.Par_AccDif;
    tmpZ = tmpDataStr.Par_PntDif;
    for i=1:4
        tmpIndex = find(tmpDataStr.CluIndex == i-1);
        plot(ax1,tmpX(tmpIndex),tmpY(tmpIndex),'Marker',Color.MkSet(i),'Color',Color.SP(k*2,:),'linestyle','none','MarkerSize',6);
        plot(ax2,tmpX(tmpIndex),tmpZ(tmpIndex),'Marker',Color.MkSet(i),'Color',Color.SP(k*2,:),'linestyle','none','MarkerSize',6);
        plot(ax3,tmpY(tmpIndex),tmpZ(tmpIndex),'Marker',Color.MkSet(i),'Color',Color.SP(k*2,:),'linestyle','none','MarkerSize',6);
        
        SelectIndex(i,k) = randsample(tmpIndex,1)
    end
    h = histogram(ax4,tmpDataStr.CluIndex,'FaceAlpha',0.2,'FaceColor',Color.SP(k*2,:));hold on;
end    
%%
close all
fig3 = figure();
set(fig3,'Color',[1,1,1],'Position',[2000 100 1200 800])
for i = 1:4
    ax(i) = subplot(2,2,i);hold on;grid on;
end

for k = 1:3
    eval(['tmpDataStr = Driver_' num2str(k) '']);
    for i=1:4
        data_index = SelectIndex(i,k);
        tmp_Data = tmpDataStr.Data{1,data_index};
%         plot(ax(i),tmp_Data(:,1),'Display','Acc Prediction','Color',Color.SP(k*2+1,:));
%         plot(ax(i),tmp_Data(:,4),'-.','Display','AccRef Prediction','Color',Color.SP(k*2+1,:));
        plot(ax(i),tmp_Data(:,6),'--','Display','Acc Measurement','Color',Color.SP(k*2+1,:));        
        plot(ax(i),tmp_Data(:,9),':','Display','AccRef Measurement','Color',Color.SP(k*2+1,:));
        plot(ax(i),tmpDataStr.Par_MaxPnt(data_index),tmp_Data(tmpDataStr.Par_MaxPnt(data_index),6),'o');
    end
end
       
    



