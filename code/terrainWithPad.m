function y = terrainWithPad(x, x_pad_center, pad_width, pad_height)
    
    terrain_fun = @(x) 1000*sin(2*pi*x/1e5) + 200*sin(10*pi*x/1e5);

    y = zeros(size(x));

    for i = 1:length(x)
        if abs(x(i) - x_pad_center) < pad_width/2
            y(i) = pad_height;  % Flat segment
        else
            y(i) = terrain_fun(x(i)); %wavy terrain 
        end
    end
end
