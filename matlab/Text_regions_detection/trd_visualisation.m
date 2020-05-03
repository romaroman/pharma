%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Description here
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

setenv('LC_ALL','C');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1) run on GALLAGER
%    matlab2014a -logfile logs/trd_quality_control_v1_S1_2.log < trd_visualisation.m  &
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
addpath(genpath('libs'));
addpath(genpath('data'));

fprintf('pid = %i\t matlab started at %s\n', feature('getpid'), datestr(now));

% --- CONSTANTS -----------------------------------------------------------
ext = '.png';

id              = 256; %343;

database        = 'Enrollment'; %'PharmaPack_R_I_S1';
srcpath         = ['../../', database, '/cropped/']; 
data_source     = ['../../', database, '/text_regions/quality_control_v1/'];

destination = 'results/19.01.2018/';

% --- Prepare data --------------------------------------------------------
createDir(destination);

files = dir([srcpath, '*', ext]);

% --- Process -------------------------------------------------------------
   
load([data_source, '/TextRegionsCoordinates_', num2str(id), '.mat']);
load([data_source, '/TextRegionsMasks_', num2str(id), '.mat']);

n = size(TextRegionsCoordinates, 1);

I = imread([srcpath files(id).name]);

fig = figure('Visible','off'); 
imshow(I, []), hold on

for i=1:n
    rect = cell2mat(TextRegionsCoordinates(i));
    drawPolygon([rect(:,1) rect(:,2)], 'b', 2); hold on;

end
saveas(fig, [destination, 'im_', num2str(id), '.png']);

% text region mask
I = double(rgb2gray(I));
for i=1:n
    mask = zeros(size(I));
    ind = TextRegionsMasks{i, 1};
    mask(ind) = 1;
    
    fig = figure('Visible','off'); 
    imshow(I.*mask, []);
    saveas(fig, [destination, 'im_', num2str(id), '_region_', num2str(i), '.png']);
end






