function [curvature] = Cal_curvature(x1, y1, x2, y2, x3, y3) 
   a = sqrt((x1-x2)^2+(y1-y2)^2); % The three sides
   b = sqrt((x2-x3)^2+(y2-y3)^2);
   c = sqrt((x3-x1)^2+(y3-y1)^2);
   A = 1/2*abs((x1-x2)*(y3-y2)-(y1-y2)*(x3-x2)); % Area of triangle
   curvature = 4*A/(a*b*c); % Curvature of circumscribing circle