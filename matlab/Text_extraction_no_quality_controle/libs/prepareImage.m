function [ gray ] = prepareImage(image)

im = double(imread(image));

if max(im(:) > 2)
    im = im / 255;
end

if size(im, 3) == 3
    gray = rgb2gray(im);
else
    gray = im;
end

clear im

end

