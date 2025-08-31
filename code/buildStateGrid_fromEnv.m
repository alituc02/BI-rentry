function [stateGrid, numStates] = buildStateGrid_fromEnv(env)
% BUILDSTATEGRID_FROMENV  Define discretization of the continuous state space.
% State = [x, h, v, gamma]
% The grid resolution depends on the training phase (A/B/C).

% === Downrange (x) bins, vary with phase ===
switch upper(env.curriculum.phase)
    case 'A'
        % Wide landing tolerance: ±500 km, coarse resolution (50 km bins)
        edges.x = linspace(env.landingZone.center - 500e3, ...
                           env.landingZone.center + 500e3, 21);
    case 'B'
        % Intermediate tolerance: ±500 km, finer resolution (50 km bins)
        edges.x = linspace(env.landingZone.center - 500e3, ...
                           env.landingZone.center + 500e3, 21);
    case 'C'
        % Strict tolerance: ±100 km, high resolution (2.5 km bins)
        edges.x = linspace(env.landingZone.center - 100e3, ...
                           env.landingZone.center + 100e3, 81);
end

% === Altitude (h) bins: higher resolution below 10 km ===
edges.h = [0 50 100 200 400 700 1000 2000 5000 10000 20000 70000];

% === Velocity (v) bins: higher resolution below 300 m/s ===
edges.v = [0 40 80 120 200 400 1000 3000 6500];

% === Flight path angle (gamma): finer resolution between -20° and 0° ===
edges.g = deg2rad([-60 -30 -20 -15 -10 -7 -5 -3 -1 0 2]);

% === Store grid info ===
stateGrid.edges = edges;
stateGrid.nx    = numel(edges.x) - 1;
stateGrid.nh    = numel(edges.h) - 1;
stateGrid.nv    = numel(edges.v) - 1;
stateGrid.ng    = numel(edges.g) - 1;
stateGrid.size  = [stateGrid.nx, stateGrid.nh, stateGrid.nv, stateGrid.ng];

% === Total number of discrete states ===
numStates = prod(stateGrid.size);
end
