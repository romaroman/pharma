%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setenv('LC_ALL','C');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1) run on GALLAGER
%    matlab2014a -logfile logs/detect_text_regions.log < detect_text_regions.m  &
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

profile on

warning off all

clear all
close all
clc

addpath(genpath('../libs'));
addpath(genpath('libs'));
addpath(genpath('data'));

fprintf('pid = %i:  Matlab started at %s\n', feature('getpid'), datestr(now));

% --- CONSTANTS -----------------------------------------------------------
is_debug         = 0;
is_visualisation = 1;
b_std            = 5;

database    = 'PharmaPack_R_I_S1';
source      = ['D:/pharmapack/', database, '/cropped/'];
destination = ['D:/pharmapack/', database, '/text_regions/no_quality_control/'];

% --- Process -------------------------------------------------------------
images = dir(source);
images  = images(~ismember({images.name},{'.','..'}));
images = {images.name}.';
n = size(images);  

% ImageList = cell(0);

fprintf('\n%s: %s start text detection \n',datestr(now), database);
l = 0;
for i=1:15
        fprintf('%s: i= %i,  %s\n', datestr(now), i, images{i});
        
        gray = prepareImage([source, images{i}]);

        [TextRegionsMasks, TextRegionsCoordinates, fig] = detectTextRegions(gray, b_std, is_debug, ...
            is_visualisation);
        
        if size(TextRegionsMasks, 1) == 0
            fprintf('\t !!! any regions have been detected\n');
            continue;
        end
        
        l = l + 1;
        % ImageList(l, :) = {images{i}};

        save([destination, '/TextRegionsMasks_', num2str(l), '.mat'],...
            'TextRegionsMasks', '-v7.3');
        save([destination, '/TextRegionsCoordinates_', num2str(l), '.mat'], ...
            'TextRegionsCoordinates', '-v7.3');

        if is_visualisation
            saveas(fig, [destination, 'visualization/TextRegions_', num2str(l), '.png']);
        end
        
        clear TextRegionsMasks
        clear TextRegionsCoordinates
        clear fig
        clear gray
        
        % save([destination, 'ListOfImages.mat'], 'ImageList');
end
profile viewer
fprintf('\n%s: %s end text detection \n',datestr(now), database);
























