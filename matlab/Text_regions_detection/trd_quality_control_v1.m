%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Text region detection
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setenv('LC_ALL','C');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1) run on GALLAGER
%    matlab2014a -logfile logs/trd_quality_control_v1.log < trd_quality_control_v1.m  &
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
nbins        = 7;
imscale      = 4;
threshold    = 0.1;
std_thr      = 10;
mask_padding = 11;

is_visualisation = 1;

database = 'Enrollment'; %'PharmaPack_R_I_S1';

srcpath         = ['D:/pharmapack/', database, '/cropped/']; 
mask_dir        = ['D:/pharmapack/', database, '/cropped/masks/'];
destination     = ['D:/pharmapack/', database, '/text_regions/quality_control_v1/'];
vis_destination = ['D:/pharmapack/', database, '/text_regions/quality_control_v1_visualisation/']; 

% --- Prepare data --------------------------------------------------------
%load([destination, 'ListOfImages.mat']);
%st = size(ImageList, 1)-1;

ImageList = cell(0);
st = 1;

files = dir([srcpath, '*', '.png']);
N     = length(files);

createDir(destination);
if is_visualisation
    createDir(vis_destination);
end

ErrID = cell(0); ll = 0;
% --- Process -------------------------------------------------------------
for n=st:N
    % results variables
    TextRegionsMasks       = cell(0);
    TextRegionsExtMasks    = cell(0);
    TextRegionsCoordinates = cell(0);

    fprintf('%s: %i %s\n', datestr(now), n, files(n).name);
    filename  = [srcpath files(n).name];
    
    ImageList(n, :) = {files(n).name};

    gray = prepareGrayImage0255(filename);
    gray = imresize(gray, 1/imscale);

    % in order to reduce the influence of lights deffects
    s = stdfilt(gray, ones(3));
    s = s > std_thr;
    
    [Gmag, Gdir] = imgradient(gray);
    Gmag = Gmag.*s;
    Gdir = Gdir.* s; 
    %--------------------------------------------
    
    % detection of edge orientations
    ind = ceil((Gdir+180) / (360/(nbins-1))) + 1;

    Gmag = Gmag / max(Gmag(:));
    binary_mask   = Gmag > threshold;

    Edges = binary_mask.*ind/nbins;

    % remove too long edges 
    ind_new = zeros(size(ind));
    for j=1:nbins
        A = ind;
        A(A ~= j) = 0;
        A(A ~= 0) = 1;
        A = A.*binary_mask;
        
        A = imclearborder(A);
        A = imdilate(A, ones(3));
        
        cc  = bwconncomp(A);
        lab = labelmatrix(cc);  
        
        props = regionprops(lab, 'Area');
        idx   = find([props.Area] < 300);
        bw    = ismember(lab, idx);   
        
        A = ind; A(A ~= j) = 0;        
        ind_new = ind_new + bw.*A;
    end
   
    Edges = binary_mask.*ind_new/nbins;

    % -------------------------------------------
    gray = prepareGrayImage0255(filename);
    
    % return to original scale
    Edges = imresize(Edges, imscale);
    if size(Edges, 1) > size(gray, 1)
        Edges = Edges(1:size(gray, 1), :);
    end
    if size(Edges, 2) > size(gray, 2)
        Edges = Edges(:, 1:size(gray, 2));
    end
    
    % mask obtained during image cropping in order to reduce the influence
    % of border between the residual background and package
    mask = ones(size(Edges));
    if ~isempty(mask_dir)
        mask = prepareGrayImage0255([mask_dir, 'mask_', files(n).name]);
        mask(mask ~= 0) = 1;
    end   
    
    Edges = Edges.* mask;
    bw    = Edges; % binary mask
    bw(bw ~= 0) = 1;

    % Text detection procedure
    bw = imdilate(bw, ones(3));
    bw = imfill(bw, 'holes');
    cc = bwconncomp(bw);
    lab = labelmatrix(cc);

    props = regionprops(lab, 'Solidity', 'Area', 'MinorAxisLength');
    idx = find( [props.Solidity] > 0.33 & [props.Area] > 1550 &  [props.MinorAxisLength] > 20);
    bw = ismember(lab, idx);
    
    scale = 1;
    len = 10*scale;
    bw2 = imresize(bw, scale);
    
    theta = -90:15:90-15;
    for k = 1:length(theta)
        b = imdilate(bw2, strel('line', len, theta(k)) );
        count(k) = sum(double(b(:)));
    end
    [~,k] = min(count);
    
    b = imdilate(bw2, strel('line', len/2, theta(k)) );
    bin = imresize(b, 1/scale);
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    cc  = bwconncomp(bin);
    lab = labelmatrix(cc);

    if is_visualisation
        fig = figure('Visible','off'); 
        imshow(gray, []), hold on
    end    

    l = 0;
    for k=1:max(max(lab))
        blob = lab==k;
        [y x] = find(blob);
        [rect, angle, area] = brect([x y]);
        rect = round(rect);

        % save detection region indices
        BM = roipoly(gray, round(rect(:, 1)), round(rect(:, 2))).*mask;
        ind = find(BM(:) == 1);
        
        if isempty(ind)
            continue;
        end
        
        l = l + 1;
        TextRegionsMasks(l, 1) = {ind};
        TextRegionsMasks(l, 2) = {size(ind, 1)};
        
        % extend mask in order to avoid border effect 
        BM_ext = extendMask(BM, mask_padding).*mask;       
        ind = find(BM_ext(:) == 1);
        TextRegionsExtMasks(l, 1) = {ind};
        TextRegionsExtMasks(l, 2) = {size(ind, 1)};
        
        TextRegionsCoordinates(l, 1) = {rect};
        
        if is_visualisation
            drawPolygon([rect(:,1) rect(:,2)], 'b', 2); hold on;
        end  
        clear BM_ext;
        clear BM;
    end
    
    if l == 0
        fprintf('\n\n any text regions \n\n');
        ll = ll + 1 ;
        ErrID(ll, :) = {n, {files(n).name}};
        save([destination, 'ListOfErrIDs.mat'], 'ErrID');
    end

    save([destination, '/TextRegionsMasks_', num2str(n), '.mat'],...
        'TextRegionsMasks', '-v7.3');
    save([destination, '/TextRegionsExtMasks_', num2str(n), '.mat'],...
        'TextRegionsExtMasks', '-v7.3');    
    save([destination, '/TextRegionsCoordinates_', num2str(n), '.mat'], ...
        'TextRegionsCoordinates', '-v7.3');

    save([destination, 'ListOfImages.mat'], 'ImageList');

    if is_visualisation
        saveas(fig, [vis_destination, 'TextRegions_', num2str(n), '.png']);
    end    
    
    fig.Visible = true;
    clear TextRegionsMasks
    clear TextRegionsExtMasks
    clear TextRegionsCoordinates 
    
end
fprintf('\n\n%s: %s End\n', datestr(now));
