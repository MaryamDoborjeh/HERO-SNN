% plot_hero_consensus_FX.m - per-neuron summary of HERO-SNN refinement over the
% trained reservoirs of the fixed-parameter run (one snapshot per fold, seed 1):
%  (a) mean depression received per neuron: summed weight reduction of the
%      excitatory connections entering the neuron, normalised within reservoir,
%      averaged over reservoirs
%  (b) consistency: fraction of reservoirs in which the neuron is among the 20%
%      most depressed
%  (c) mean homeostatic threshold change (theta after - theta before);
%      orange = raised, teal = lowered
% Black circles = input neurons. Run in the folder holding the snapshot files.
clear; clc; close all;

FOLDS = 1:10;                       % E2: the 10 patient folds
TOPFRAC = 0.20;

nF = numel(FOLDS); dep = []; thr = [];
for k = 1:nF
    S = load(sprintf('hero_snapshot_all_full_FX_f%d.mat', FOLDS(k)));
    if k == 1
        P = S.coords; n = size(P,1);
        isInt = true(n,1); isInt(S.inputIdx) = false; isInt(S.shadow) = false;
        inp = S.inputIdx; dep = zeros(n, nF); thr = zeros(n, nF);
    end
    D = max(S.W0 - S.W1, 0); D(~(S.plastic & S.W0 > 0)) = 0;
    d = sum(D, 1)';                              % depression entering each neuron
    dep(:,k) = d / max(d(isInt));                % normalise within reservoir
    th0 = S.th0(:); th1 = S.th1(:);
    if numel(th0) == 1, th0 = th0 * ones(n,1); end
    thr(:,k) = th1 - th0;
end
meanDep = mean(dep, 2);
cons = zeros(n,1);
for k = 1:nF
    v = dep(isInt, k); cut = quantile(v, 1 - TOPFRAC);
    cons = cons + (dep(:,k) >= cut & isInt);
end
cons = cons / nF;
meanThr = mean(thr, 2);

fprintf('%d reservoirs | mean threshold change: raised %d, lowered %d of %d internal neurons\n', ...
    nF, nnz(meanThr(isInt) > 0), nnz(meanThr(isInt) < 0), nnz(isInt));

figure(1); clf; set(gcf, 'Position', [20 20 1650 560], 'Color', 'w');
x = P(:,1); y = P(:,2); z = P(:,3);
ttl = {sprintf('(d) Mean depression received per neuron\n(normalised within reservoir, averaged over %d reservoirs)', nF), ...
       sprintf('(e) Consistency: fraction of reservoirs in which the neuron\nis among the top %d%% depressed', round(100*TOPFRAC)), ...
       sprintf('(f) Mean homeostatic threshold change\n(orange = raised, teal = lowered)')};
BG  = [0.82 0.82 0.82];                % light grey: neurons with no change (shows the brain shape)
RED = [0.75 0.1 0.1]; BLUE = [0.1 0.3 0.8];
for panel = 1:3
    subplot(1,3,panel); hold on;
    scatter3(x(isInt), y(isInt), z(isInt), 12, BG, 'filled');
    if panel == 1
        v = meanDep; a = v / max(v(isInt));
        m = isInt & a > 0; col = BG + (RED - BG) .* a(m);     % grey -> red
        scatter3(x(m), y(m), z(m), 10 + 60*a(m), col, 'filled');
    elseif panel == 2
        v = cons; a = v;
        m = isInt & a > 0; col = BG + (BLUE - BG) .* a(m);    % grey -> blue
        scatter3(x(m), y(m), z(m), 10 + 60*a(m), col, 'filled');
    else
        v = meanThr; a = abs(v) / max(abs(v(isInt)) + eps);
        m = isInt & a > 0; vi = v(m); ai = a(m); col = zeros(nnz(m), 3);
        up = vi > 0;
        col(up,:)  = 1 - (1 - [0.9 0.5 0.1]) .* ai(up);
        col(~up,:) = 1 - (1 - [0.1 0.5 0.5]) .* ai(~up);
        scatter3(x(m), y(m), z(m), 10 + 60*ai, col, 'filled');
    end
    scatter3(x(inp), y(inp), z(inp), 40, 'k', 'LineWidth', 1);
    title(ttl{panel}, 'FontSize', 9); axis equal; axis tight; axis off; view(-35, 25);
    % colour code
    t = linspace(0, 1, 64)';
    if panel < 3
        if panel == 1, cmap = BG + (RED - BG) .* t; else, cmap = BG + (BLUE - BG) .* t; end
        colormap(gca, cmap); caxis([0 1]); cb = colorbar('southoutside');
        if panel == 1, xlabel(cb, 'normalised depression received (0 = none, 1 = max)', 'FontSize', 8);
        else, xlabel(cb, sprintf('fraction of %d reservoirs in top %d%%', nF, round(100*TOPFRAC)), 'FontSize', 8); end
    else
        lim = max(abs(meanThr(isInt))) + eps;
        cmap = [1 - (1 - [0.1 0.5 0.5]) .* flipud(t); 1 - (1 - [0.9 0.5 0.1]) .* t];   % teal -> white -> orange
        colormap(gca, cmap); caxis([-lim lim]); cb = colorbar('southoutside');
        xlabel(cb, 'mean threshold change \theta_j (after - before)', 'FontSize', 8);
    end
    set(cb, 'FontSize', 8);
end
print(gcf, 'hero_consensus_E2_FX.png', '-dpng', '-r200');
fprintf('saved hero_consensus_E2_FX.png\n');
