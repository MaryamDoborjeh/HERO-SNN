% plot_snn_connections_FX.m - 3D view of reservoir connectivity from a fixed-run snapshot:
% (a) after STDP (W0), (b) after HERO-SNN (W1), (c) synapses depressed by HERO (W0 - W1).
% Blue = excitatory, red = inhibitory; line width scales with |weight|.
% Black dots = reservoir neurons, yellow circles = input neurons.
clear; clc; close all;

FOLD     = 3;        % fold to draw (E1: 2,3,4,5,6,7,8,10,12,13; E2: 1..10)
NTRAIN   = 168;      % training epochs per pass in this fold (E1 = 168; E2 = 720)
DRAWFRAC = 1.0;      % 1.0 = all connections; 0.5 = strongest half

snapFile = sprintf('hero_snapshot_all_full_FX_f%d.mat', FOLD);
S = load(snapFile);
P = S.coords; n = size(P,1);
isInt = true(n,1); isInt(S.inputIdx) = false; isInt(S.shadow) = false;

nConn  = nnz(S.conn & S.W0 ~= 0);               % all connections after STDP
nAfter = nnz(S.conn & S.W1 ~= 0);               % non-zero connections after HERO
pos    = S.plastic & (S.W0 > 0);                % excitatory internal (plastic) synapses
nPos   = nnz(pos); nDep = nnz(pos & S.W1 < S.W0); nZero = nnz(pos & S.W1 == 0);
fires  = S.heroFires(:)';

fprintf('fold %d: %d connections after STDP, %d non-zero after HERO-SNN\n', FOLD, nConn, nAfter);
fprintf('        %d excitatory internal synapses: %d depressed, %d driven to zero\n', nPos, nDep, nZero);
fprintf('        misclassified training presentations per pass: %s of %d (eta = %g, passes = %d)\n', ...
    mat2str(fires), NTRAIN, S.eta, S.passes);

figure(1); clf; set(gcf, 'Position', [20 20 1650 520], 'Color', 'w');
ttl = {sprintf('(a) After STDP training (baseline), fold %d\n%d connections (blue = excitatory, red = inhibitory)', FOLD, nConn), ...
       sprintf('(b) After HERO-SNN refinement\n%d connections (blue = excitatory, red = inhibitory)', nAfter), ...
       sprintf('(c) Difference: synapses depressed by HERO\n%d synapses depressed (%d driven to 0)', nDep, nZero)};
for panel = 1:3
    subplot(1,3,panel); hold on;
    if panel == 1, W = S.W0; elseif panel == 2, W = S.W1; else, W = S.W0 - S.W1; end
    if panel < 3
        M = S.conn & (abs(W) > 0);
    else
        M = S.conn & (abs(W) > 1e-12);
    end
    [ii, jj] = find(M);
    w = W(M);
    if DRAWFRAC < 1 && panel < 3
        th = quantile(abs(w), 1 - DRAWFRAC);
        keep = abs(w) >= th; ii = ii(keep); jj = jj(keep); w = w(keep);
    end
    aw = abs(w) / (max(abs(w)) + eps);
    for pass = 1:2
        if pass == 1, sel = aw <= 0.4; lw = 0.2; else, sel = aw > 0.4; lw = 0.8; end
        if ~any(sel), continue; end
        if panel < 3
            selP = sel & (w > 0); selN = sel & (w < 0);
        else
            selP = false(size(sel)); selN = sel;      % depression drawn in red
        end
        for k = 1:2
            if k == 1, s = selP; col = [0 0.2 0.9]; else, s = selN; col = [0.9 0.1 0.1]; end
            if ~any(s), continue; end
            X = [P(ii(s),1) P(jj(s),1) nan(nnz(s),1)]';
            Y = [P(ii(s),2) P(jj(s),2) nan(nnz(s),1)]';
            Z = [P(ii(s),3) P(jj(s),3) nan(nnz(s),1)]';
            plot3(X(:), Y(:), Z(:), '-', 'Color', col, 'LineWidth', lw);
        end
    end
    plot3(P(isInt,1), P(isInt,2), P(isInt,3), '.', 'Color', [0 0 0], 'MarkerSize', 6);
    plot3(P(S.inputIdx,1), P(S.inputIdx,2), P(S.inputIdx,3), 'o', ...
        'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', [1 0.85 0], 'MarkerSize', 6);
    title(ttl{panel}, 'FontSize', 9); xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal; axis tight; view(-35, 25); grid on; box on;
end
outFile = sprintf('snn_connections_FX_f%d.png', FOLD);
print(gcf, outFile, '-dpng', '-r200');
fprintf('saved %s\n', outFile);
