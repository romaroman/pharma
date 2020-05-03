function [min_rect, angle, min_area] = brect(points)

psize = size(points); %razmernostta na spisuka

points_ind = convhull(points(:,1), points(:,2));

points_ind(length(points_ind), :) = []; %iztrivame posledniq index.
%obichainia nachin e poslednata tochka na polygon-a da se povtarq
%za da bude toi zatvoren. Naprimer, ako imame tochki
%1,2,3,4,5,6,7,8 i se okaje che 2, 3, 5, 7, 8 obrazuvat polygon-a
%convhull vrushta 2, 3, 5, 7, 8, 2.

valid_points = points(points_ind, :); %polzvaiki indexite,
%ostavqme samo tochkite, koito ni trqbvat.

%prilagame algorituma na Dennis S. Arnon---------------

n = length(valid_points); %broi tochki

%centura na mnogougulnika (za da vartim okolo nego)
cx = sum(valid_points(:,1)) / n;
cy = sum(valid_points(:,2)) / n;

%inicializirame
min_area = inf;
min_rect = zeros(4,2);

for i=1:n
    next = i + 1;
    if (i == n) %tai kato polzvame purvata tochka pri smetkite na poslednata
        next = 1;
    end
    
    if (valid_points(i,1) == valid_points(next,1))
        current_angle = 0.5 * pi;
    else
        current_angle = atan((valid_points(next, 2) - valid_points(i,2)) / (valid_points(next,1) - valid_points(i,1)));
    end
    
    %da poluchim koordinatite na zavurtqnite tochki okolo centara na
    %mnogougulnika
    [points_rotated_x, points_rotated_y] = rotateP(valid_points(:,1), valid_points(:,2), cx, cy, -current_angle);
    
    %vzimame krainite tochki
    left = min(points_rotated_x);
    right = max(points_rotated_x);
    bottom = min(points_rotated_y);
    top = max(points_rotated_y);
    current_area = abs(left-right) * abs(top-bottom);
    
    %zapisvame tochkite vav vid na pravougulnik
    rect = [left bottom; right bottom; right top; left top];
    
    %zavurtame pravougulnika obratno no okolo centara na MNOGOUGULNIKA,
    %ne okolo negovia si center.
    
    [rect_x rect_y] = rotateP(rect(:,1), rect(:,2), cx, cy, +current_angle);
    
    if (current_area < min_area)
        min_area = current_area;
        min_rect = [rect_x rect_y];
        angle = current_angle;
    end
    
end

%da narisuvame pravougulnika otnovo
% draw_polygon(valid_points, 'k');
% draw_polygon([min_rect(:,1) min_rect(:,2)], 'b');

function [X, Y] = rotateP(X, Y, cx, cy, a)
%zavurtane na polygon s tochki X, Y (vektori) na ugul a okolo tochka (cx,cy)

    %translirame
    Xt = X - cx;
    Yt = Y - cy;
    
    %vurtim
    Xr = Xt * cos(a) - Yt * sin(a);
    Yr = Xt * sin(a) + Yt * cos(a);
    
    %translirame obratno
    X = Xr + cx;
    Y = Yr + cy;
    
