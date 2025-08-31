function idx = discretize_state(state, stateGrid)
% DISCRETIZE_STATE  Map continuous state values into discrete grid indices.
% State = [x, h, v, gamma]
%
% Returns a linear index into the Q-table (sub2ind).

x = state(1); 
h = state(2); 
v = state(3); 
g = state(4);

% Downrange: if grid has only 1 bin (x excluded), force ix=1
if numel(stateGrid.edges.x) == 2
    ix = 1;
else
    ix = discretize_clamped(x, stateGrid.edges.x);
end

% Altitude, velocity, flight path angle
ih = discretize_clamped(h, stateGrid.edges.h);
iv = discretize_clamped(v, stateGrid.edges.v);
ig = discretize_clamped(g, stateGrid.edges.g);

% Convert subscript indices → single linear index
idx = sub2ind(stateGrid.size, ix, ih, iv, ig);
end


function ii = discretize_clamped(val, edges)
% DISCRETIZE_CLAMPED  Like discretize(), but clamps values outside edges.
%
% Inputs:
%   val   - scalar or vector value(s)
%   edges - monotonically increasing vector of bin edges (length ≥ 2)
%
% Output:
%   ii    - bin index in [1, numel(edges)-1]

    ii = discretize(val, edges);   % may return NaN for values outside range
    nanmask = isnan(ii);
    if any(nanmask(:))
        below = val < edges(1);
        ii(nanmask & below)  = 1;                 % clamp below first edge
        ii(nanmask & ~below) = numel(edges) - 1;  % clamp above last edge
    end
end
