function logs = train_sarsa(env, episodes, max_steps)
% TRAIN_SARSA  Tabular SARSA with GLIE, optimistic Q-init, and logging.
% Requirements:
%   - Environment created with createReentryEnvironment
%   - Functions: buildStateGrid_fromEnv, discretize_state, stepReentry,
%                computeReward, epsilonGreedy, checkTermination

if nargin < 2, episodes  = 3000; end
if nargin < 3, max_steps = 400;  end

% === Build state grid ===
[stateGrid, numStates] = buildStateGrid_fromEnv(env);

% === Action space ===
if strcmpi(env.action.type,'absolute')
    nA = numel(env.action.bankAngles_abs);
else
    nA = numel(env.action.bankAngles_inc);
end

% === Q-table: optimistic initialization ===
Q = 50 * ones(numStates, nA);

% === Hyperparameters (inspired by Lunar Lander) ===
alpha    = 0.01;   % fixed learning rate
% alpha0   = 0.4;   % alternative: initial learning rate for Robbins–Monro schedule
gammaRL  = 0.99;    % fixed discount factor
eps0     = 1;       % initial epsilon
eps_min  = 0.001;   % minimum epsilon
eps_decay= 0.995;    % aggressive exponential decay

% === Logging structures ===
logs.success_rate = zeros(episodes,1);
logs.is_success   = false(episodes,1);
logs.final_v      = nan(episodes,1);
logs.final_dx     = nan(episodes,1);
logs.reasons      = strings(episodes,1);
logs.total_reward = zeros(episodes,1);   % cumulative reward per episode
logs.steps        = zeros(episodes,1);   % steps executed per episode

success_count = 0;
%Nsa = zeros(numStates, nA);   % state-action visit counter

for ep = 1:episodes
    
    % Epsilon decay 
    epsilon = max(eps_min, eps0 * (eps_decay^(ep-1)));

    % === Episode reset ===
    [env, s] = env.reset(env);   % continuous state = [x; h; v; gamma; sigma]
    s_idx    = discretize_state(s, stateGrid);
    a        = epsilonGreedy(Q, s_idx, epsilon);

    terminalReason = "running";
    ep_reward = 0; 
    traj = nan(max_steps, 5);    % trajectory log: [x, h, v, gamma, sigma]

    for t = 1:max_steps
        % --- Step dynamics ---
        [s2, done, reason] = stepReentry(s, env, a);
        traj(t,:) = s2(:)';   

        % Discretize next state
        s2_idx = discretize_state(s2, stateGrid);

        % Choose next action a' using ε-greedy
        a2 = epsilonGreedy(Q, s2_idx, epsilon);

        % Compute reward (with potential-based shaping)
        r = computeReward(s2, env, s, reason);
        ep_reward = ep_reward + r;

        % === SARSA update ===
        %Nsa(s_idx, a) = Nsa(s_idx, a) + 1;

        % --- Robbins–Monro schedule (disabled here) ---
        % alpha = alpha0 / (1 + 0.05 * Nsa(s_idx, a));
        % This decreases α with visit count, ensuring convergence.

        % --- Fixed α update (chosen) ---
        td_target = r + gammaRL * Q(s2_idx, a2) * (~done);
        td_error  = td_target - Q(s_idx, a);
        Q(s_idx, a) = Q(s_idx, a) + alpha * td_error;

        % Check termination
        if done
            terminalReason = reason;
            s = s2; s_idx = s2_idx; a = a2; %#ok<NASGU>
            break;
        end

        % Transition
        s     = s2;
        s_idx = s2_idx;
        a     = a2;
    end

    % === Episode logging ===
    dx = abs(s(1) - env.landingZone.center);
    logs.final_v(ep)       = s(3);
    logs.final_dx(ep)      = dx;
    logs.reasons(ep)       = terminalReason;
    logs.total_reward(ep)  = ep_reward;
    logs.steps(ep)         = t;
    logs.trajectory{ep}    = traj(1:t,:);   % truncate to actual length

    % Success counter
    if terminalReason == "landed"
        success_count = success_count + 1;
        logs.is_success(ep) = true;
    end
    logs.success_rate(ep) = success_count / ep;

    % === Print diagnostics every 10 episodes ===
    if mod(ep, 10) == 0
        fprintf(['Ep %d | eps=%.3f | alpha=%.4f | a=%d | reason=%s | steps=%d | ' ...
                 'reward=%.1f | h=%.1f m | v=%.3f m/s | gamma=%.2f deg | x=%.1f m | SR=%.4f\n'], ...
            ep, epsilon, alpha, a, terminalReason, t, ep_reward, ...
            s(2), s(3), rad2deg(s(4)), s(1), logs.success_rate(ep));
    end
end

% === Output additional info ===
logs.Q = Q;
logs.stateGrid = stateGrid;
end
