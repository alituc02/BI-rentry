function r = computeReward(state, env, prevState, terminalReason)
% COMPUTEREWARD  Reward shaping for reentry RL environment.
%
% Components:
%   - Potential-based shaping (progress towards target conditions)
%   - Terminal rewards/penalties (success/failure cases)

    r = env.reward.stepCost;

    % === Potential-based shaping ===
    Phi_prev = 0;
    if nargin >= 3 && ~isempty(prevState)
        Phi_prev = potential(prevState, env);
    end
    Phi_curr = potential(state, env);
    r = r + env.shaping.lambda * (Phi_curr - Phi_prev);  % lambda = gammaRL

    % === Optional shaping (currently disabled) ===
    % h = state(2); v = state(3);
    % % Altitude/velocity gates (give reward for safe descent profile)
    % if h < 10000 && v < 300, r = r + 1; end
    % if h <  5000 && v < 200, r = r + 2; end
    % if h <  1000 && v < 120, r = r + 3; end
    %
    % % Heating / g-load penalties
    % [~,~,qdot,n_load] = env.compute_forces(env, max(h,0), max(v,0), state(5));
    % if env.limits.applyHeating
    %     r = r - 0.01 * (qdot / 1e5);   % scaled heating penalty
    % end
    % if env.limits.applyGload
    %     r = r - 0.05 * max(0, n_load - 1);  % penalty above 1 g
    % end

    % === Terminal rewards/penalties ===
    if nargin >= 4
        switch terminalReason
            case "landed"
                r = r + env.reward.successBonus;

            case "out_of_pad"
                dx_out = abs(state(1) - env.landingZone.center) ...
                         - env.terminal.padTolerance_m;   % overshoot distance [m]
                r = r - (dx_out/1e3)^1.4;                % nonlinear penalty

            case "crashed_fast"
                v_star = env.shaping.v_star;             % desired touchdown speed
                v      = state(3);
                dv     = max(0, v - v_star);             % excess speed
                r = r + env.reward.failPenalty ...
                      - 50 * (dv);                  % additional penalty scaled by speed

            case {"overheat","g_exceed","crashed","skipout","out_of_domain"}
                r = r + env.reward.failPenalty;
        end
    end
end


function Phi = potential(s, env)
% POTENTIAL  Shaping potential function for SARSA/QLearning.
% Lower values correspond to states closer to target.

    h     = max(s(2),0);
    v     = max(s(3),0);
    g     = s(4); %#ok<NASGU>
    dx    = abs(s(1) - env.landingZone.center);

    % Reference target velocity
    v_star = env.shaping.v_star;  % [m/s]

    % Normalization scales
    Hn = 10000;        % altitude scale [m]
    Vn = 50;           % velocity scale [m/s]
    Gn = deg2rad(10);  % gamma deviation scale [rad] (not used here)
    Sn = deg2rad(60);  % bank angle scale [rad]    (not used here)
    Xn = 100e3;        % downrange scale [m]

    % Weighted sum of penalties (negative potential)
    Phi = -( ...
        env.shaping.wh * (h/Hn) ...
      + env.shaping.wv * abs(v - v_star)/Vn ...
      + env.shaping.wx * (dx/Xn) ...
    );
    % + env.shaping.wg * abs(g - g_star)/Gn ... % currently disabled
    % + env.shaping.ws * abs(sigma)/Sn ...      % currently disabled
end
