KH_ColorCode

x = 1:100;
y = 10;
Hsize = 600;
Wsize = 700;
figure('name','PlotColor','numbertitle','off')
set(gcf,'Color',[1,1,1],'position',[0 1000-Hsize Wsize Hsize])
set(gca,'linewidth',0.7,'FontSize',5)
%================
subplot(5,2,[1 2])
hold on
for i=1:29
    bar(x(i),y,'facecolor',CP(i,:))
end
title('Color Code (CP)')
set(gca,'xtick',[1:29])
hold off
%================
subplot(5,2,3)
hold on
for i=1:10
    bar(x(i),y,'facecolor',RP(i,:))
end
title('Red Code Map (RP)')
set(gca,'xtick',[1:10],'xlim',[0 11])
%================
subplot(5,2,4)
hold on
for i=1:10
    bar(x(i),y,'facecolor',YP(i,:))
end
title('Yellow Code Map (YP)')
set(gca,'xtick',[1:10],'xlim',[0 11])
%================
subplot(5,2,5)
hold on
for i=1:10
    bar(x(i),y,'facecolor',GP(i,:))
end
title('Green Code Map (GP)')
set(gca,'xtick',[1:10],'xlim',[0 11])
%================
subplot(5,2,6)
hold on
for i=1:10
    bar(x(i),y,'facecolor',BP(i,:))
end
title('Blue Code Map (BP)')
set(gca,'xtick',[1:10],'xlim',[0 11])
%================
subplot(5,2,7)
hold on
for i=1:10
    bar(x(i),y,'facecolor',PP(i,:))
end
title('Purple Code Map (PP)')
set(gca,'xtick',[1:10],'xlim',[0 11])
%================
subplot(5,2,8)
hold on
for i=1:10
    bar(x(i),y,'facecolor',WP(i,:))
end
title('White Code Map (WP)')
set(gca,'xtick',[1:10],'xlim',[0 11])
%================
subplot(5,2,9)
hold on
for i=1:10
    bar(x(i),y,'facecolor',SP(i,:))
end
title('Selected (SP)')
set(gca,'xtick',[1:10],'xlim',[0 11])


%%
