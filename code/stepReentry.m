function [next_state, done, reason] = stepReentry(current_state, env, action_idx)
% STEP_REENTRY  One integration step of planar point-mass reentry dynamics.
% State vector: [x; h; v; gamma; sigma]
%
% Inputs:
%   current_state = [downrange, altitude, velocity, flight path angle, bank angle]
%   env           = environment structure (forces, atmosphere, vehicle, etc.)
%   action_idx    = index of chosen control action
%
% Outputs:
%   next_state    = updated state after dt
%   done          = true if terminal condition reached
%   reason        = string with termination reason

% === Unpack current state ===
x = current_state(1);
h = current_state(2);
v = current_state(3);
g = current_state(4);          % flight path angle [rad]
sigma_prev = current_state(5);

% === 1) Actuator dynamics (rate limit, delay, noise) ===
sigma = env.apply_action(env, sigma_prev, action_idx);

% === 2) Aerodynamic forces and atmosphere ===
[L, D, ~, ~] = env.compute_forces(env, max(h,0), max(v,0), sigma);

% === 3) Physical parameters ===
r    = env.R_E + h;
grav = env.gravity(r);                 % gravitational acceleration [m/s^2]
m    = env.vehicle.mass;
dt   = env.dt;
wind = env.domain.wind_bias;

% === 4) Stability tricks for gamma integration ===
v_eff = max(v, 300);                   % avoid division by near-zero velocity

% === 5) Point-mass derivatives ===
dh = v * sin(g);
dx = v * cos(g) + wind;
dv = -D/m - grav * sin(g);
%dg = (L_eff/(m * v_eff)) + (v * cos(g) / r) - (grav * cos(g) / v_eff);
dg = (L/(m * v)) + (v * cos(g) / r) - (grav * cos(g) / v);

% Limit angular rate to avoid jitter
%dg = max(min(dg, deg2rad(10)), deg2rad(-10));  % clamp to ±10°/s

% === 6) Euler integration ===
x2 = x + dx * dt;
h2 = h + dh * dt;
v2 = max(0, v + dv * dt);
g2 = g + dg * dt;
g2 = max(min(g2, deg2rad(85)), deg2rad(-85));  % clamp gamma to ±85°

% === 7) Build next state ===
next_state = [x2; h2; v2; g2; sigma];

% === 8) Check termination ===
[done, reason] = checkTermination(next_state, env);

end
