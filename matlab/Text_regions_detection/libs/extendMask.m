function [ mask ] = extendMask(mask, padding)

mask_bottom  = padarray(mask, [padding, 0], 'post');
mask_bottom(1:padding, :) = [];           
mask = mask + mask_bottom;

mask_left  = padarray(mask, [0, padding], 'pre');
mask_left(:, end-padding+1:end) = [];             
mask = mask + mask_left;

mask_top  = padarray(mask, [padding, 0], 'pre');
mask_top(end-padding+1:end, :) = [];
mask = mask + mask_top;

mask_right  = padarray(mask, [0, padding], 'post');
mask_right(:, 1:padding) = [];        
mask = mask + mask_right;

mask(mask ~= 0) = 1;
end

