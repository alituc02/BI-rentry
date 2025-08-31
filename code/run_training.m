%% RUN_TRAINING.M  — Quick start for SARSA training + diagnostic plots

clear; clc; close all;

% 1) Create environment (choose training phase A/B/C and action type)
cfg = struct();
cfg.phase = 'B';                % 'B' => wide pad ±50 km, heating/g-load limits
cfg.dt    = 0.2;                % integration time step
cfg.action_type = 'absolute';   % 'absolute' | 'incremental'
env = createReentryEnvironment(cfg);

% 2) Train agent (number of episodes and maximum steps per episode)
episodes  = 10000;
max_steps = 3500;
logs = train_sarsa(env, episodes, max_steps);

% Extract last trajectory and corresponding time vector
traj = logs.trajectory{end};
time = (0:size(traj,1)-1) * env.dt;

% === Plot trajectory (x, h, v) ===
figure;
subplot(3,1,1);
plot(time, traj(:,1)/1000); grid on;
ylabel('x [km]');

subplot(3,1,2);
plot(time, traj(:,2)/1000); grid on;
ylabel('h [km]');

subplot(3,1,3);
plot(time, traj(:,3)); grid on;
ylabel('v [m/s]'); xlabel('Time [s]');

% === Plot success rate ===
W = 200;  % moving average window
logs.sr_ma = movmean(double(logs.is_success), W, 'Endpoints','shrink');

figure;
plot(1:episodes, logs.sr_ma, 'LineWidth', 1.8); hold on;
plot(1:episodes, logs.success_rate, '--', 'LineWidth', 1.2);
grid on; xlim([1 episodes]); ylim([0 1]);
legend(sprintf('SR moving avg (W=%d)', W), 'Cumulative SR', 'Location', 'southeast');
xlabel('Episode'); ylabel('Success probability');
title('Training success rate');

% === Plot total reward (raw + moving average) ===
figure;
hold on;
plot(1:episodes, logs.total_reward, ':', 'Color',[0.6 0.6 0.6]);     % raw reward
plot(1:episodes, movmean(logs.total_reward,200),'LineWidth',1.8);   % moving average
grid on;
xlabel('Episode'); ylabel('Reward');
title('Reward per episode');
legend('Raw reward','200-ep moving avg','Location','best');

% === Plot final downrange error ===
figure;
plot(1:episodes, movmean(logs.final_dx,200),'LineWidth',1.8);
grid on; xlabel('Episode'); ylabel('|x - pad| [m]');
title('Final downrange error');

% === Plot final touchdown velocity ===
figure;
plot(1:episodes, movmean(logs.final_v,200),'LineWidth',1.8);
grid on; xlabel('Episode'); ylabel('Velocity [m/s]');
title('Final touchdown velocity');

% === Histogram of termination reasons ===
figure;
histogram(categorical(logs.reasons));
grid on;
xlabel('Termination reason');
ylabel('Count');
title('Distribution of termination reasons');

% 3) (Optional) save training results
save('sarsa_logs.mat', 'logs');
disp('Training completed and logs saved in sarsa_logs.mat');
