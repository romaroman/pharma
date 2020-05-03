function [ I ] = prepareImage0255(image)

I = imread(char(image));

if (length(size(I)) > 2)
    I = rgb2gray(I);
end

I = im2double(I);

end
