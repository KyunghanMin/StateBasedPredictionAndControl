%점의 좌표를 입력

Radius = [];
RoadPos= 1450:1900;
for i  = RoadPos(1):RoadPos(end)
    x1=X(i);    
    y1=Y(i);
    x2=X(i+1);   
    y2=Y(i+1);
    
    
    x3=X(i+2);
    
    y3=Y(i+2);
    
    
    
    %두 수직 이등분선의 기울기를 입력
    
    d1=(x2-x1)/(y2-y1);
    
    d2=(x3-x2)/(y3-y2);
    
    
    
    %원의 중점을 구함
    
    cx=((y3-y1)+(x2+x3)*d2-(x1+x2)*d1)/(2*(d2-d1));
    
    cy=-d1*(cx-(x1+x2)/2)+(y1+y2)/2;
    
    %원의 반지름을 구함
    
    
    
    r=sqrt((x1-cx)^2+(y1-cy)^2);
    input=[x1 y1; x2 y2; x3 y3];
    [R,xyrc] = fit_circle_through_3_points(input);
    Radius=[Radius R];
    
    
end
figure(99)
plot(RoadPos,Radius);
ylim([0,100]);

filtered_curvature = movmean(TwoDim_CurvatureRoadModel,5);
figure(98)
plot( filtered_curvature);

d_curvature = [];
for i = 1 : length(filtered_curvature)-1
    d_curvature = [d_curvature, filtered_curvature(i+1)-filtered_curvature(i)];
end

figure(97)
plot(d_curvature)

%% Reymond lateral acc margin model

FirstCornerPos=find(RoadLength>650 & RoadLength<780);
figure(96)
plot(VelocityPerRoad(FirstCornerPos))
d_c_max = max(d_curvature(FirstCornerPos));


figure(95)
plot(X,Y,'r', 'linewidth', 3);
hold on;
plot(X(FirstCornerPos),Y(FirstCornerPos),'go' ,'linewidth', 3);
grid on;
