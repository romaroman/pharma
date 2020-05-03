function [ TextRegionsMasks, TextRegionsCoordinates, fig ] = detectTextRegions(gray, b_std, is_debug, ...
    is_visualisation)

TextRegionsMasks       = cell(0);
TextRegionsCoordinates = cell(0);  

% --- extract text ----
s = stdfilt(gray, ones(b_std));

th = graythresh(s);
bw = s > th;

if is_debug
    figure; imshow(bw, []);
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
bw = imclearborder(bw);
bw = imdilate(bw, ones(b_std));
bw = imfill(bw, 'holes');

if is_debug
    figure; imshow(bw, []);
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
cc = bwconncomp(bw);
lab = labelmatrix(cc);
if is_debug
    figure; imshow(lab, []);
end        
props = regionprops(lab, 'Solidity', 'Area', 'MinorAxisLength');
idx = find( [props.Solidity] > 0.33 & [props.Area] > 150 & [props.MinorAxisLength] > 20);
bw = ismember(lab, idx);

scale = 1/4;
len = 200*scale;
bw2 = imresize(bw, scale);


theta = -90:15:90-15;
for k = 1:length(theta)
    b = imdilate(bw2, strel('line', len, theta(k)) );
    count(k) = sum(double(b(:)));
end
[~,k] = min(count);

b = imdilate(bw2, strel('line', len/2, theta(k)) );
bin = imresize(b, 1/scale);

% --- save results ----
fig = '';
if is_visualisation
    if is_debug
        fig = figure;
    else
        fig = figure('Visible','off'); 
    end

    imshow(gray), hold on
end
cc = bwconncomp(bin);
lab = labelmatrix(cc);
ij = 0;
for k=1:max(max(lab))
    blob = lab==k;
    [y x] = find(blob);
    [rect, angle, area] = brect([x y]);
    if is_visualisation
        drawPolygon([rect(:,1) rect(:,2)], 'b', 2);
    end

    BM = roipoly(gray, round(rect(:, 1)), round(rect(:, 2)));
    [ind, ~] = find(BM(:) == 1);

    ij = ij + 1;
    TextRegionsMasks(ij,1)     = {ind};
    TextRegionsMasks(ij,2)     = {size(ind, 1)};
    TextRegionsCoordinates(ij) = {rect};               
end

end

