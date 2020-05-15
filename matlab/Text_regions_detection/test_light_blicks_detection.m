%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Description here
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

setenv('LC_ALL','C');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1) run on GALLAGER
%    matlab2014a -logfile logs/log.log < test_light_blicks_detection.m  &
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

addpath(genpath('libs'));
addpath(genpath('data'));

fprintf('pid = %i\t matlab started at %s\n', feature('getpid'), datestr(now));

% --- CONSTANTS -----------------------------------------------------------
dateFormat  = 'dd.mm.yyyy';

nbins = 27;

database    = 'PharmaPack_R_I_S1';

source      = ['D:/pharmapack/', database, '/cropped/']; 
destination = ['results/test_light_blicks_detection/', datestr(now,dateFormat), '/'];


% --- Prepare data --------------------------------------------------------
createDir(destination);

files = dir([source, '*.png']);
N     = length(files);

% --- Process -------------------------------------------------------------
for i=1:N%[2, 60, 61, 84, 105, 164,  292, 361]%1:N
    to = destination;%[destination, num2str(i), '/']; createDir(to);
    
%     fprintf('%s: %i %s\n', datestr(now), i, files(i).name);
fprintf('%i %s\n', i, files(i).name);

    Irgb = imread([source files(i).name]);
    
   fig = figure('Visible','off'); 
   imshow(Irgb, []);
   saveas(fig, [to, 'img_', num2str(i), '.png']);     
% continue    
    Ihsv = rgb2hsv(Irgb); 
    H    = Ihsv(:, :, 1);
    
%     fig = figure('Visible','off'); 
%     imshow(H, []); colorbar;
%     title(sprintf('%0.5f - %0.5f : %0.5f', min(H(:)), max(H(:)), max(H(:))-min(H(:))));
%     saveas(fig, [to, 'img_', num2str(i), '_h.png']); 
    %{
    [counts,centers] = hist(H(:), nbins);
    C = [counts; 1:nbins]';
    C = flipud(sortrows(C));
    x = C(2, 1) / C(1, 1);
    
    fig = figure('Visible','off'); 
    hist(H(:), 64);
    title(sprintf('%i, %i : %0.3f', C(1, 1), C(2, 1), x));
    saveas(fig, [to, 'img_', num2str(i), '_Hist.png']);     
   %}
    hs = stdfilt(H, ones(3));
    hs = hs > 0.02;

%     fig = figure('Visible','off'); 
%     imshow(hs, []); colorbar;
%     saveas(fig, [to, 'img_', num2str(i), '_Hmask.png']);      

    mx = max(H(:));
    hi = abs(H - mx);
%     fig = figure('Visible','off'); 
%     imshow(hi, []); colorbar;
%     saveas(fig, [to, 'img_', num2str(i), '_Hi.png']);  
    
    hi = hi > 0.1;
    fig = figure('Visible','off'); 
    imshow(hi, []); colorbar;
    saveas(fig, [to, 'img_', num2str(i), '_Himask.png']);       
end























