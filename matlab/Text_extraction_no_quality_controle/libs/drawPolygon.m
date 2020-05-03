function drawPolygon(poly, gfx, lw)
    poly(end+1, 1) = poly(1,1);
    poly(end, 2) = poly(1,2);
    plot(poly(:, 1), poly(:, 2), 'color', gfx, 'linewidth', lw); 
