%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Description here
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

setenv('LC_ALL','C');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1) run on GALLAGER
%    matlab2014a -logfile logs/log.log < test_light_blicks_detection_HL.m  &
%
% run VL_FEAT:
% 1)GALLAGER
%    run('/opt/matlab_external/vlfeat-0.9.20/toolbox/vl_setup');
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off all;

close all
clear
clc

addpath(genpath('../libs'));
addpath(genpath('../images'));
addpath(genpath('libs'));
addpath(genpath('data'));

fprintf('pid = %i\t matlab started at %s\n', feature('getpid'), datestr(now));

% --- CONSTANTS -----------------------------------------------------------
dateFormat  = 'dd.mm.yyyy';

nbins = 27;

database    = 'PharmaPack_R_I_S1';

source      = ['../../', database, '/cropped/']; 
destination = ['results/test_light_blicks_detection_HL/', datestr(now,dateFormat), '/'];


% --- Prepare data --------------------------------------------------------
createDir(destination);

files = dir([source, '*.png']);
N     = length(files);

% --- Process -------------------------------------------------------------
for i=[2 20 53 60 61 76 84 105 164 292 361]%1:N
    to = [destination, num2str(i), '/']; createDir(to);
    
    fprintf('%s: %i %s\n', datestr(now), i, files(i).name);

    Irgb = double(imread([source files(i).name]));
    
    fig = figure('Visible','off'); 
    imshow(Irgb, []);
    saveas(fig, [to, 'img_', num2str(i), '.png']); 
    
    M = [0.412453 0.357580 0.180423
          0.212671 0.715160 0.072169
          0.019334 0.119193 0.950227];
    YM = M(2, :);
    
    R = Irgb(:, :, 1);
    G = Irgb(:, :, 2);
    B = Irgb(:, :, 3);
      
    Y = M(2, :)* [R(:)'; G(:)'; B(:)'];
    
    ybm1 = Y > 0.008856;
    ybm2 = ~ybm1;
    
    Y1 = Y.*ybm1; 
    Y1 = 116*(Y1.^(1/3))-16;
    Y2 = Y.*ybm2;
    Y2 = 903.3*Y2;
    
    L = reshape(Y1+Y2, size(R));
    
    fig = figure('Visible','off'); 
    imshow(L, []);
    saveas(fig, [to, 'img_', num2str(i), '_l.png']);       
    
    % H
    Ihsv = rgb2hsv(Irgb);
    H = Ihsv(:, :, 1);
    
    fig = figure('Visible','off'); 
    imshow(H, []);
    saveas(fig, [to, 'hsv_img_', num2str(i), '_h.png']);   
    %{
    
    d = max(Irgb, [], 3);
    H = 60*(G-B)./d;
    
    fig = figure('Visible','off'); 
    imshow(H, []);
    saveas(fig, [to, 'img_', num2str(i), '_h.png']);  
    %}
    
    HL = H - L;
    fig = figure('Visible','off'); 
    imshow(HL, []); colorbar;
    saveas(fig, [to, 'img_', num2str(i), '_hl.png']);    
continue
    %{
    % lab 
    colorTransform = makecform('srgb2lab');
    Ilab = double(applycform(Irgb, colorTransform));
    
    L = Ilab(:, :, 1);
    
    fig = figure('Visible','off'); 
    imshow(L, []);
    saveas(fig, [to, 'lab_img_', num2str(i), '_l.png']);     
   
    
    % hsv 
    Ihsv = rgb2hsv(Irgb);
    H = Ihsv(:, :, 1);
    
    fig = figure('Visible','off'); 
    imshow(H, []);
    saveas(fig, [to, 'hsv_img_', num2str(i), '_h.png']);     
    
    
    HL = H-L;
    fig = figure('Visible','off'); 
    imshow(HL, []);
    saveas(fig, [to, 'diff_img_', num2str(i), '_hl.png']);    
    
    s = stdfilt(HL, ones(3));
    fig = figure('Visible','off'); 
    imshow(s, []); colorbar;
    saveas(fig, [to, 'hl_img_', num2str(i), '_std.png']);  
    
    %{
    [counts,centers] = hist(HL(:), 256);
    A = sortrows([counts; centers]');
    
    thr = A(end, 2);
    
    bw = HL < thr;
    fig = figure('Visible','off'); 
    imshow(bw, []); colorbar;
    saveas(fig, [to, 'diff_img_', num2str(i), '_bw.png']);     
    %}
%}
   
end























