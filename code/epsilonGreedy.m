function a = epsilonGreedy(Q, s_idx, epsilon)
if rand < epsilon
    a = randi(size(Q,2));   % esplora
else
    [~, a] = max(Q(s_idx, :));  % sfrutta
end
end
