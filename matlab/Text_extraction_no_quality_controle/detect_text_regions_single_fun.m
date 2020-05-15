function [dst_path_visu, dst_path_masks, dst_path_coords] = detect_text_regions_single_fun(src_path, num)
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

warning off all

addpath(genpath('../libs'));
addpath(genpath('libs'));
addpath(genpath('data'));

fprintf('pid = %i:  Matlab started at %s\n', feature('getpid'), datestr(now));

% --- CONSTANTS -----------------------------------------------------------
is_debug         = 0;
is_visualisation = 1;
b_std            = 5;

database    = 'PharmaPack_R_I_S1';
destination = ['D:/pharmapack/', database, '/text_regions/no_quality_control/'];

% --- Process -------------------------------------------------------------
image = src_path;

fprintf('\n%s: %s start text detection \n',datestr(now), database);
fprintf('%s: i= %i,  %s\n', datestr(now), image);

gray = prepareImage(image);

[TextRegionsMasks, TextRegionsCoordinates, fig] = detectTextRegions(gray, b_std, is_debug, ...
    is_visualisation);

if size(TextRegionsMasks, 1) == 0
    fprintf('\t !!! any regions have been detected\n');
    return
end

dst_path_masks = [destination, '/TextRegionsMasks_', num2str(num), '.mat'];
save(dst_path_masks, 'TextRegionsMasks', '-v7.3');
masks_mat = TextRegionsMasks;

dst_path_coords = [destination, '/TextRegionsCoordinates_', num2str(num), '.mat'];
save(dst_path_coords, 'TextRegionsCoordinates', '-v7.3');
coords_mat = TextRegionsCoordinates;

if is_visualisation
    dst_path_visu = [destination, 'visualization/TextRegions_', num2str(num), '.png'];
    saveas(fig, dst_path_visu);
end
fprintf('\n%s: %s end text detection \n',datestr(now), database);
























