function [done, reason] = checkTermination(state, env)
% CHECKTERMINATION  Check if the episode should end and why.
%
% Terminal conditions consistent with env:
% - SUCCESS: touchdown with v <= v_max_success, inside pad tolerance
% - FAILURES: crash (high velocity impact), out_of_pad, overheating, g-limit,
%             skip-out (bounce to high altitude), out_of_domain

% === Unpack state ===
x     = state(1); 
h     = state(2); 
v     = state(3); 
gamma = state(4); % gamma not used as termination condition

% === Terrain check ===
z_terr = env.terrain(x);
touch  = (h <= z_terr + 5);     % 5 m tolerance for touchdown
dx     = abs(x - env.landingZone.center);

% === Landing pad tolerance  ===
padActive = isfield(env,'curriculum') && isfield(env.curriculum,'padActive') && env.curriculum.padActive;
if padActive
    padTol   = env.terminal.padTolerance_m;   % defined only in phase C
    inPadTol = (dx <= padTol);
else
    inPadTol = true;                          % ignore pad in phases A/B
end

% === Success thresholds ===
v_ok = v <= env.terminal.v_max_success;

% === Aerothermal and g-load limits ===
[~, ~, qdot, n_load] = env.compute_forces(env, max(h,0), max(v,0), state(5));
overheat = env.limits.applyHeating && (qdot > env.heating.q_max);
gexceed  = env.limits.applyGload   && (n_load > env.limits.n_max);

% === Skip-out condition (reentry failure) ===
skipout = (h >= env.skip.h_thr) && (gamma > env.skip.gamma_min);

% === Terminal cases ===
if h < -100 || v < 0
    done = true;  reason = "out_of_domain"; return;
end

if overheat
    done = true;  reason = "overheat"; return;
end

if gexceed
    done = true;  reason = "g_exceed"; return;
end

if skipout
    done = true;  reason = "skipout"; return;
end

% --- Touchdown cases ---
if touch
    if v_ok && inPadTol
        done = true;  reason = "landed"; return;
    elseif v_ok && ~inPadTol
        done = true;  reason = "out_of_pad"; return;
    elseif v > env.terminal.v_max_success
        done = true;  reason = "crashed_fast"; return;
    else
        done = true;  reason = "crashed"; return;
    end
end

% === Otherwise, keep running ===
done = false; 
reason = "running";
end
