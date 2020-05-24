function [ imCropped, mask, imCropped_rgb] = cropPackage( image, padding, tresh )

mask          = [];
imCropped     = [];
imCropped_rgb = [];

% --- Prepare data --------------------------------------------------------
I    = im2double(rgb2gray(imread(image)));
Irgb = im2double(imread(image));

% --- Process -------------------------------------------------------------
% binarize the image
imBW = I > tresh;

% extract characteristics of the regions
stats = regionprops(imBW,'Area','ConvexImage','BoundingBox','Centroid', 'Orientation');

% take the region with the biggest area
allArea = [stats.Area];
[~, maxId] = max(allArea);

imCropped = imcrop(I,stats(maxId).BoundingBox);
[mh, mw]  = size(imCropped);
mask      = stats(maxId).ConvexImage;
mask      = imresize(mask,[mh, mw]);

n = length(size(Irgb));
if n > 2
    for i=1:n
        imCropped_rgb(:, :, i) = imcrop(Irgb(:, :, i),stats(maxId).BoundingBox);
    end
end

% --- narrow mask in order to avoid border effect -------------------------
mask_top  = padarray(mask, [padding, 0], 'pre');
mask_top(end-padding+1:end, :) = [];
mask_new = mask.*mask_top;

mask_left  = padarray(mask, [0, padding], 'pre');
mask_left(:, end-padding+1:end) = [];
mask_new = mask_new.*mask_left;

mask_bot  = padarray(mask, [padding, 0], 'post');
mask_bot(1:padding, :) = [];
mask_new = mask_new.*mask_bot;

mask_right  = padarray(mask, [0, padding], 'post');
mask_right(:, 1:padding) = [];
mask_new = mask_new.*mask_right;

mask = mask_new;


end

