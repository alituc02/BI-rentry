function env = createReentryEnvironment(cfg)
% CREATEENTRYENVIRONMENT  RL environment for planar point-mass reentry.
%
% Features:
%   - Curriculum phases A/B/C (loose → strict pad, progressive constraints)
%   - Randomization of initial conditions and environment per episode
%   - Absolute or incremental bank angle actions with actuator model
%   - Realistic success/failure thresholds for touchdown
%
% Usage:
%   env = createReentryEnvironment();                        % default
%   env = createReentryEnvironment(struct('phase','B','dt',0.2));
%
% Reset episode:
%   [env, s0] = env.reset(env);  % s0 = [x; h; v; gamma; sigma]

%% === Physical constants & atmosphere ===
env.g0   = 9.80665;
env.R_E  = 6371e3;
env.mu   = 3.986004418e14;
env.atmosphere.H    = 7200;
env.atmosphere.rho0 = 1.225;
env.models.rho = @(env,h) max(env.atmosphere.rho0 * env.domain.rho_scale .* ...
                              exp(-max(h,0)./env.atmosphere.H), 0);

%% === Integration time step ===
env.dt   = GET(cfg,'dt',0.2);

%% === Landing zone & terrain ===
env.landingZone.center   = GET(cfg,'pad_center',0);         
env.landingZone.width    = GET(cfg,'pad_width_terrain',100e3);
env.landingZone.altitude = 0;

env.terrainType = GET(cfg,'terrainType','flat');
switch env.terrainType
    case 'flat'
        env.terrain = @(x) zeros(size(x));
    case 'mountain'
        env.terrain = @(x) terrainWithPad(x, env.landingZone.center, ...
                                          env.landingZone.width, ...
                                          env.landingZone.altitude);
    otherwise
        error('Unrecognized terrainType.');
end

%% === Curriculum (A/B/C) ===
env.curriculum.phase = GET(cfg,'phase','A');
switch upper(env.curriculum.phase)
    case 'A'
        env.curriculum.padActive   = true;
        env.terminal.padTolerance_m = 50e3;   
        env.limits.applyHeating     = false;
        env.limits.applyGload       = false;
    case 'B'
        env.curriculum.padActive   = true;
        env.terminal.padTolerance_m = 50e3;    
        env.limits.applyHeating     = true;
        env.limits.applyGload       = true;
    case 'C'
        env.curriculum.padActive   = true;
        env.terminal.padTolerance_m = 10e3;  
        env.limits.applyHeating     = true;
        env.limits.applyGload       = true;
    otherwise
        error('Phase must be A, B, or C.');
end

%% === Vehicle / Aerodynamics ===
env.vehicle.mass_nom   = GET(cfg,'mass',3000);   % [kg]
env.vehicle.S_ref      = GET(cfg,'S',30);        % [m^2]
env.vehicle.CD_nom     = GET(cfg,'CD',1.05);
env.vehicle.CL_nom     = GET(cfg,'CL',0.35);
env.vehicle.sigma_min  = deg2rad(-85);
env.vehicle.sigma_max  = deg2rad( 85);

env.models.CD = @(env) env.vehicle.CD_nom * env.domain.CD_scale;
env.models.CL = @(env) env.vehicle.CL_nom * env.domain.CL_scale;

%% === Actions: bank angle ===
env.action.type = GET(cfg,'action_type','absolute');  % 'absolute' | 'incremental'

% Absolute set (recommended for early training)
env.action.bankAngles_abs = deg2rad([0, -45, -30, -15, 15, 30, 45]);

% Incremental actions
env.action.bankAngles_inc = deg2rad([-15, -7, 0, 7, 15]);

% Actuator dynamics
env.action.rateLimit   = deg2rad(GET(cfg,'rateLimit_deg',20));  
env.action.noiseStd    = deg2rad(GET(cfg,'noise_deg',0.5));     
env.action.delaySteps  = GET(cfg,'delay_steps',1);              
env.action.sigma0      = 0.0;                                  

%% === Episode randomization ===
% Initial conditions (entry interface range)
env.icRange.h0_m   = GET(cfg,'h0_range',     [66e3, 68e3]);
env.icRange.g0_rad = deg2rad(GET(cfg,'g0_range_deg', [-6, -4]));
env.icRange.v0_ms  = GET(cfg,'v0_range',     [5.9e3, 6.1e3]);


% Domain randomization ranges
env.domain.rho_scale_range   = [0.95, 1.05];
env.domain.CD_scale_range    = [0.95, 1.05];
env.domain.CL_scale_range    = [0.9,  1.1];
env.domain.mass_scale_range  = [0.98, 1.02];
env.domain.wind_bias_range   = [-20, 20];   % [m/s]

% Initialize with nominal values (reset will randomize)
env.domain.rho_scale  = 1.0;
env.domain.CD_scale   = 1.0;
env.domain.CL_scale   = 1.0;
env.domain.mass_scale = 1.0;
env.domain.wind_bias  = 0.0;

%% === Thermal / g-load / skip constraints ===
env.heating.k        = GET(cfg,'heating_k', 1.83e-4);  
env.heating.q_max    = GET(cfg,'q_max', 800e3);        

env.limits.n_max     = GET(cfg,'n_max', 8);            

env.skip.h_thr       = 70e3;                           
env.skip.gamma_min   = 0;                              

%% === Touchdown success thresholds ===
env.terminal.h_max_success     = 50;    % [m]
env.terminal.v_max_success     = 40;    % [m/s]
% gamma bounds no longer used

%% === Reward bonuses/penalties (env-level) ===
env.reward.stepCost        = 0;      
env.reward.successBonus    = 100;        
env.reward.failPenalty     = -250;      

% Shaping weights (used in computeReward.m)
env.shaping.v_star     = 40;              
env.shaping.gamma_star = deg2rad(-6);     
env.shaping.wv = 0.7;
env.shaping.wg = 0;     
env.shaping.wh = 0.2;
env.shaping.ws = 0;     
env.shaping.wd = 0.1;   
env.shaping.wx = 2;     
env.shaping.lambda = 0.995;

%% === Utility functions exposed ===
env.reset          = @resetEpisode;     
env.apply_action   = @applyAction;      
env.compute_forces = @computeForces;    
env.gravity        = @(r) env.mu./(r.^2);

end % ==== createReentryEnvironment =======================================


%% =======================================================================
function [env, s0] = resetEpisode(env)
% RESETEPISODE  Randomize domain and initial conditions, return initial state.
rng('shuffle');

% Reset actuator buffer
env.apply_action(env, NaN, "reset");

% Domain randomization
env.domain.rho_scale  = unif(env.domain.rho_scale_range);
env.domain.CD_scale   = unif(env.domain.CD_scale_range);
env.domain.CL_scale   = unif(env.domain.CL_scale_range);
env.domain.mass_scale = unif(env.domain.mass_scale_range);
env.domain.wind_bias  = unif(env.domain.wind_bias_range);

% Initial state
h0 = unif(env.icRange.h0_m);
v0 = unif(env.icRange.v0_ms);
g0 = unif(env.icRange.g0_rad);
if g0 > deg2rad(-2), g0 = deg2rad(-3.5); end

% Crude ballistic range estimate to place start before the pad
k    = 0.75;
range_est = h0 * cot(abs(g0));
x0   = -k * range_est + 20e3*(2*rand-1); 
x0   = min(max(x0, -800e3), -50e3);

sigma0 = env.action.sigma0;

% Effective mass per episode
env.vehicle.mass = env.vehicle.mass_nom * env.domain.mass_scale;

% Initial state vector
s0 = [x0; h0; v0; g0; sigma0];
end


%% =======================================================================
function sigma_next = applyAction(env, sigma_prev, action_idx)
% APPLYACTION  Apply action with actuator dynamics: delay, rate-limit, noise.
% If action_idx == 'reset' → reset delay buffer and return sigma0.

persistent delayLine

% --- Reset buffer (used at env.reset) ---
if (nargin >= 3) && ( (ischar(action_idx) && strcmp(action_idx,'reset')) ...
                   || (isstring(action_idx) && action_idx=="reset") )
    delayLine = [];
    sigma_next = env.action.sigma0;
    return;
end

if isempty(delayLine)
    delayLine = zeros(max(1, env.action.delaySteps), 1);
end

switch lower(env.action.type)
    case 'absolute'
        sigma_cmd = env.action.bankAngles_abs(action_idx);
    case 'incremental'
        d = env.action.bankAngles_inc(action_idx);
        sigma_cmd = sigma_prev + d;
    otherwise
        error('Unrecognized action.type.');
end

% Saturation
sigma_cmd = min(max(sigma_cmd, env.vehicle.sigma_min), env.vehicle.sigma_max);

% Delay (FIFO)
if env.action.delaySteps > 0
    sigma_delayed = delayLine(end);
    delayLine = [sigma_cmd; delayLine(1:end-1)];
else
    sigma_delayed = sigma_cmd;
end

% Rate limit (DISABLED here for simplicity)
% ds = sigma_delayed - sigma_prev;
% ds = max(min(ds, env.action.rateLimit), -env.action.rateLimit);
% sigma_limited = sigma_prev + ds;
sigma_limited = sigma_delayed;

% Actuator noise
sigma_noisy = sigma_limited + env.action.noiseStd * randn;

% Final saturation
sigma_next = min(max(sigma_noisy, env.vehicle.sigma_min), env.vehicle.sigma_max);
end


%% =======================================================================
function [L, D, qdot, n_load] = computeForces(env, h, v, sigma)
% COMPUTEFORCES  Compute Lift, Drag, heating rate, and aerodynamic g-load.
rho = env.models.rho(env,h);             
CD  = env.models.CD(env);
CL  = env.models.CL(env);
q   = 0.5 * rho .* v.^2;                 

D = q .* env.vehicle.S_ref .* CD;
L = q .* env.vehicle.S_ref .* CL .* cos(sigma);  

% Heating model (simplified Sutton–Graves-like)
qdot = env.heating.k .* sqrt(rho) .* v.^3;

% Aerodynamic g-load (L&D norm) normalized by g0
n_load = sqrt(L.^2 + D.^2) ./ (env.vehicle.mass * env.g0);
end


%% =======================================================================
function y = unif(range2)
% UNIF  Uniform random sample in [a,b].
y = range2(1) + (range2(2)-range2(1))*rand;
end


%% =======================================================================
function val = GET(S, field, default)
% GET  Get struct field with default fallback.
    if nargin < 3, default = []; end
    if isempty(S) || ~isstruct(S) || ~isfield(S, field) || isempty(S.(field))
        val = default;
    else
        val = S.(field);
    end
end


%% =======================================================================
% function env = set_randomization_level(env, level)
% % SET_RANDOMIZATION_LEVEL  Example utility to interpolate randomization.
%     level = max(0,min(1,level));
%     env.icRange.h0_m   = blend_range(env.rnd.tight.ic.h0_m,   env.rnd.wide.ic.h0_m,   level);
%     env.icRange.v0_ms  = blend_range(env.rnd.tight.ic.v0_ms,  env.rnd.wide.ic.v0_ms,  level);
%     env.icRange.g0_rad = blend_range(env.rnd.tight.ic.g0_rad, env.rnd.wide.ic.g0_rad, level);
% 
%     env.domain.rho_scale_range  = blend_range(env.rnd.tight.dom.rho_scale,  env.rnd.wide.dom.rho_scale,  level);
%     env.domain.CD_scale_range   = blend_range(env.rnd.tight.dom.CD_scale,   env.rnd.wide.dom.CD_scale,   level);
%     env.domain.CL_scale_range   = blend_range(env.rnd.tight.dom.CL_scale,   env.rnd.wide.dom.CL_scale,   level);
%     env.domain.mass_scale_range = blend_range(env.rnd.tight.dom.mass_scale, env.rnd.wide.dom.mass_scale, level);
%     env.domain.wind_bias_range  = blend_range(env.rnd.tight.dom.wind_bias,  env.rnd.wide.dom.wind_bias,  level);
% 
%     env.rnd.level = level;
% end
% 
% function r = blend_range(a, b, t)
% % BLEND_RANGE  Linear interpolation of [a,b] intervals.
%     r = [ a(1) + t*(b(1)-a(1)),  a(2) + t*(b(2)-a(2)) ];
% end
