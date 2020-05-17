%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Description here
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

setenv('LC_ALL','C');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1) run on GALLAGER
%    matlab2014a -logfile logs/log.log < test_light_blicks_detection_and_edges.m  &
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
nbins        = 7;
imscale      = 4;
threshold    = 0.075; %0.1;
std_thr      = 3; % = 10;  to avoid long edges

is_debug   = 0;
dateFormat = 'dd.mm.yyyy';

database = 'PharmaPack_R_I_S1';

srcpath         = ['D:/pharmapack/', database, '/cropped/']; 
mask_dir        = ['D:/pharmapack/', database, '/cropped/masks/'];
save_to      = ['results/test_light_blicks_detection_and_edges/', datestr(now,dateFormat), '_v3/'];

% --- Prepare data --------------------------------------------------------
createDir(save_to);

files = dir([srcpath, '*.png']);
N     = length(files);

% --- Process -------------------------------------------------------------
for n=[60 61 76 100 145 146 250 251 292]%1:N%[2, 4 5 1 13 60, 61, 84, 105, 164,  292, 361]%63%[60 61]%1:N

    fprintf('%s: %i %s\n', datestr(now), n, files(n).name);
    filename  = [srcpath files(n).name];

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
    
    % saveImage(Edges, [save_to, 'Edges_', files(n).name]);
    % continue

    if is_debug | 1
        fig = figure('Visible','off'); 
        imshow(Edges, []); colorbar;
        set(gcf, 'ColorMap', colormap([0 0 0; hsv(nbins)]));
        saveas(fig, [save_to, 'img_', num2str(n), 'e1.png']); 
    end    
    
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

    if is_debug
        fig = figure('Visible','off'); 
        imshow(Edges, []); colorbar;
        set(gcf, 'ColorMap', colormap([0 0 0; hsv(nbins)]));
        saveas(fig, [save_to, 'img_', num2str(n), 'e2.png']); 
    end    
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

    if is_debug
        fig = figure('Visible','off'); 
        imshow(Edges, []); colorbar;
        set(gcf, 'ColorMap', colormap([0 0 0; hsv(nbins)]));
        saveas(fig, [save_to, 'img_', num2str(n), 'e3.png']); 
    end    
    
    %-------------------
    Irgb = imread(filename);
    
    if is_debug
        fig = figure('Visible','off'); 
        imshow(Irgb, []); colorbar;
        saveas(fig, [save_to, 'img_', num2str(n), '.png']);     
    end
    
    Ihsv = rgb2hsv(Irgb); 
    H    = Ihsv(:, :, 1);
    
    s = stdfilt(H, ones(5));
    s = s > 0.01;
    s = imdilate(s, ones(5));
    if is_debug
        fig = figure('Visible','off'); 
        imshow(s, []); colorbar;
        saveas(fig, [save_to, 'img_', num2str(n), 'std.png']); 
    end 
    
    H = H .* s;
    if is_debug
        fig = figure('Visible','off'); 
        imshow(H, []); colorbar;
        saveas(fig, [save_to, 'img_', num2str(n), 'h.png']); 
    end 
    
    mx = max(H(:));
    hi = abs(H - mx);
    hi = hi + s;
    hi = hi > 0.1;
    if is_debug
        fig = figure('Visible','off'); 
        imshow(hi, []); colorbar;
        saveas(fig, [save_to, 'img_', num2str(n), 'hi.png']); 
    end    
    
    Edges = Edges .* hi;
    
    if is_debug
        fig = figure('Visible','off'); 
        imshow(Edges, []); colorbar;
        set(gcf, 'ColorMap', colormap([0 0 0; hsv(nbins)]));
        saveas(fig, [save_to, 'img_', num2str(n), 'e4.png']); 
    end

    %-------------------    
   
    Edges = Edges.* mask;
    bw    = Edges; % binary mask
    bw(bw ~= 0) = 1;

    % Text detection procedure
    bw = imdilate(bw, ones(3));
    bw = imfill(bw, 'holes');
    cc = bwconncomp(bw);
    lab = labelmatrix(cc);

    props = regionprops(lab, 'Solidity', 'Area', 'MinorAxisLength');
    idx = find( [props.Solidity] > 0.33 & [props.Area] > 1000 &  [props.MinorAxisLength] > 20);
%     idx = find( [props.Solidity] > 0.33 & [props.Area] > 1000);
    bw = ismember(lab, idx);
    
    if is_debug
        fig = figure('Visible','off'); 
        imshow(bw, []); colorbar;
        set(gcf, 'ColorMap', colormap([0 0 0; hsv(nbins)]));
        saveas(fig, [save_to, 'img_', num2str(n), 'bw.png']); 
    end
    
    
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

    fig = figure('Visible','off'); 
    imshow(gray, []), hold on

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

        drawPolygon([rect(:,1) rect(:,2)], 'b', 2); hold on;

    end
    saveas(fig, [save_to, 'TextRegions_', num2str(n), '.png']);

    
end
