function [ mask ] = lightMask( im )

m2 = max(imdilate(im,ones(3)),[],3);

k = 0.85;
bw2 = m2 > k*max(max(m2));

ed = edge(m2);
ed = bwareaopen(ed, 10);
ed = imdilate(ed, ones(15));

mask = ~(~ed & bw2);

end

