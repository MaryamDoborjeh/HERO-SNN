%% plot_hero_weights_pool_FX.m - HERO refinement, two panels (works for E1 and E2), fixed-parameter run:
%%   (a) per-synapse weight before vs after HERO (positive internal synapses;
%%       below the diagonal = depressed) - read directly from the snapshot
%%   (b) number of eligible synapses across training samples (replayed from the
%%       snapshot's BEFORE network with the fold's eta/passes/drift from Table 3(b))
%% Run in the folder of a COMPLETED run (Havening or NIH - auto-detected).
%% No checkpoints are touched. Panel (b) replays the seed-1 Phase-3 pass, so it
%% is a mechanism illustration (seed-1 networks), like the other change figures.
clear; clc; close all;

FOLD = 3;                                   % which fold's snapshot to use

%% ---- auto-detect dataset / configuration ----
if isfile('hero_dataset_havening_64ch_2s.mat')
    dataFile = 'hero_dataset_havening_64ch_2s.mat';
    trainPerClass = 7;  cycleOn = false;  dsName = 'Havening';
elseif isfile('hero_dataset_nih10_6ch_05s.mat')
    dataFile = 'hero_dataset_nih10_6ch_05s.mat';
    trainPerClass = 40; cycleOn = true;   dsName = 'NIH seizure';
else
    error('no known dataset file in this folder');
end
snapFile = sprintf('hero_snapshot_all_full_FX_f%d.mat', FOLD);
snap = load(snapFile);

% fold's Phase-3 settings from the fixed-run results file (fallback: edit here)
eta = 0.01; passes = 1; drift = 0.00256;
if isfile('prepost_final_results_all_full_FX.mat')
    RR = load('prepost_final_results_all_full_FX.mat');
    r = RR.results{FOLD};
    if isfield(r,'eta'), eta = r.eta; passes = r.passes; drift = r.drift; end
end
if isfield(snap,'eta'), eta = snap.eta; passes = snap.passes; end
fprintf('%s | fold %d | replaying with eta=%.3g, passes=%d, drift=%.5g\n', ...
    dsName, FOLD, eta, passes, drift);

% constants (as the driver)
cfg.leak=0.002; cfg.refractory=6; cfg.stdpRate=0.01; cfg.roMod=0.8;
cfg.aerAlpha=0.5; cfg.K=1; cfg.sigma=1; cfg.condVel=3.5;
cfg.homeoRate=0.01; cfg.homeoTarget=0.05; cfg.threshMin=0.1; cfg.threshMax=1.5;
cfg.cycleChannels = cycleOn; cfg.drift = drift;

%% ---- rebuild the BEFORE network from the snapshot (as the transmission meter) ----
S = load(dataFile); X = double(S.X); y = S.y(:); pid = S.pid(:);
n1 = numel(snap.th0);
R = struct('W', snap.W0, 'thresh', snap.th0, 'conn', snap.conn, 'plastic', snap.plastic, ...
    'inputIdx', snap.inputIdx(:), 'shadow', snap.shadow(:)', 'n', n1, ...
    'tGlobal', 0, 'lastSpk', zeros(1, n1), 'stdpRec', zeros(n1));
D = zeros(n1); P = snap.coords;
for k = 1:3, D = D + (P(:,k) - P(:,k)').^2; end
D = sqrt(D); Dly = max(1, round(D / cfg.condVel)); Dly(~snap.conn) = 0;
dv = unique(Dly(snap.conn))';
R.dlyVals = dv; R.maxDly = max([dv, 1]);
R.DlyMask = cell(1, numel(dv));
for k = 1:numel(dv), R.DlyMask{k} = (Dly == dv(k)); end

isTest = pid == FOLD;
trainIdx = subsample_train(find(~isTest), y, pid, trainPerClass);
lab = y(trainIdx); nTr = numel(trainIdx);
allP = unique(pid)'; thP = cell(max(allP),1);
for p = allP, thP{p} = aer_threshold(X(:,:,pid == p), cfg.aerAlpha); end
fprintf('fold %d: %d training samples | snapshot %s\n', FOLD, nTr, snapFile);

% prototypes under the BEFORE network
W = zeros(nTr, n1);
fprintf('propagating prototypes...\n');
for i = 1:nTr
    [sp, sn] = encode_sample(X(:,:,trainIdx(i)), thP{pid(trainIdx(i))}, cfg);
    W(i,:) = desnn_weights(propagate(sp, sn, R, cfg, false, false), cfg);
end

%% ---- replay Phase 3 (depression-only Eq. 12 + homeostasis), logging the pool ----
poolLog = zeros(1, passes*nTr); actLog = poolLog; errLog = false(1, passes*nTr); k = 0;
for pass = 1:passes
    for i = 1:nTr
        [raster, R, elig, rates] = propagate(SPP(X, trainIdx(i), thP, pid, cfg), ...
                                             SNN(X, trainIdx(i), thP, pid, cfg), R, cfg, false, true); %#ok<*NASGU>
        if max(elig(:)) > 0, elig = elig / max(elig(:)); end
        wi = desnn_weights(raster, cfg);
        others = true(nTr,1); others(i) = false;
        yh = wknn_predict(wi, W(others,:), lab(others), cfg.K, cfg.sigma);
        k = k + 1;
        poolLog(k) = nnz(R.plastic & (R.W > 0));             % plastic positive pool
        actLog(k)  = nnz(R.plastic & (R.W > 0) & (elig > 0));% eligible on THIS sample
        errLog(k)  = (yh ~= lab(i));
        if errLog(k)                                          % error -> depress (Eq. 12)
            mask = R.plastic & (R.W > 0);
            R.W(mask) = max(R.W(mask) - eta*elig(mask), 0);
        end
        adj = cfg.homeoRate * (rates - cfg.homeoTarget);      % homeostasis: every sample
        adj(R.inputIdx) = 0; adj(R.shadow) = 0;
        R.thresh = min(max(R.thresh + adj, cfg.threshMin), cfg.threshMax);
    end
    if pass < passes                                          % prototype recomputation
        for i = 1:nTr
            [sp, sn] = encode_sample(X(:,:,trainIdx(i)), thP{pid(trainIdx(i))}, cfg);
            W(i,:) = desnn_weights(propagate(sp, sn, R, cfg, false, false), cfg);
        end
    end
end
% sanity: how close did the replay land to the saved AFTER network?
excM = snap.conn & snap.plastic & (snap.W0 > 0);
fprintf('replay check: ||W_replay - W1|| / ||W1 - W0|| = %.3f (small = faithful)\n', ...
    norm(R.W(excM) - snap.W1(excM)) / max(norm(snap.W1(excM) - snap.W0(excM)), eps));
fprintf('eligible pool: %d -> %d synapses | errors (updates): %d of %d samples\n', ...
    poolLog(1), poolLog(end), nnz(errLog), k);

%% ---- figure ----
figure(1); clf; set(gcf, 'Position', [40 40 1150 470], 'Color', 'w');
BLU = [0.17 0.35 0.72];

subplot(1,2,1);                                               % (a) weight scatter
excM = snap.conn & snap.plastic & (snap.W0 > 0);
e0 = snap.W0(excM); e1 = snap.W1(excM);
plot(e0, e1, '.', 'Color', [0.28 0.40 0.65], 'MarkerSize', 5); hold on;
lim = [0, max(e0)*1.05];
plot(lim, lim, 'k--');
axis([lim lim]); axis square;
xlabel('weight BEFORE HERO'); ylabel('weight AFTER HERO');
title(sprintf('(a) Positive internal synapses (n = %d)\nbelow diagonal = depressed (%d; %d driven to 0)', ...
    numel(e0), nnz(e1 < e0), nnz(e1 == 0)));

subplot(1,2,2);                                               % (b) eligible-pool decline
plot(1:k, poolLog(1:k), '-', 'Color', BLU, 'LineWidth', 1.1); hold on;
ee = find(errLog(1:k));
plot(ee, poolLog(ee), '.', 'Color', [0.80 0.25 0.20], 'MarkerSize', 8);
if passes > 1
    for pb = 1:passes-1
        plot([pb*nTr pb*nTr]+0.5, ylim, ':', 'Color', [0.55 0.55 0.55]);
    end
end
legend({'plastic positive synapses', 'error -> update applied'}, 'Location', 'northeast');
xlabel('Training presentation index'); ylabel('Number of eligible synapses');
title(sprintf('(b) Eligible synapse pool across training presentations\n%d -> %d (updates on %d of %d presentations)', ...
    poolLog(1), poolLog(k), numel(ee), k));

print('-dpng', '-r200', sprintf('hero_weights_pool_FX_f%d.png', FOLD));
fprintf('Saved hero_weights_pool_FX_f%d.png\n', FOLD);

%% ================= helpers (as the run driver / transmission meter) =================
function sp = SPP(X, ix, thP, pid, cfg)
[sp, ~] = encode_sample(X(:,:,ix), thP{pid(ix)}, cfg);
end
function sn = SNN(X, ix, thP, pid, cfg)
[~, sn] = encode_sample(X(:,:,ix), thP{pid(ix)}, cfg);
end

function [raster, R, elig, rates] = propagate(sp, sn, R, cfg, doSTDP, doElig)
T = size(sp,1); n = R.n;
flag = zeros(1,n); pot = zeros(1,n); refr = zeros(1,n);
nB = R.maxDly + 1; B = zeros(nB, n);
raster = zeros(T, n);
needE = (nargout >= 3) && doElig;
if needE, elig = zeros(n); else, elig = []; end
rate = cfg.stdpRate * 0.1;
for t = 1:T
    R.tGlobal = R.tGlobal + 1; te = R.tGlobal;
    ind_sp = (flag == 1);
    flag(:) = 0;
    ready = (refr == 0);
    if any(ind_sp)
        R.lastSpk(ind_sp) = te;
        for kD = 1:numel(R.dlyVals)
            a = mod(t - 2 + R.dlyVals(kD), nB) + 1;
            B(a, :) = B(a, :) + sum(R.W(ind_sp, :) .* R.DlyMask{kD}(ind_sp, :), 1);
        end
    end
    cur = mod(t - 1, nB) + 1;
    pot(ready) = pot(ready) + B(cur, ready);
    B(cur, :) = 0;
    fire = pot >= R.thresh';
    flag(fire) = 1;
    pot(fire) = 0; refr(fire) = cfg.refractory;
    pot(~fire) = max(0, pot(~fire) - cfg.leak);
    refr(~fire) = max(0, refr(~fire) - 1);
    if any(fire) && (doSTDP || needE)
        I = find(fire);
        dt = te - R.lastSpk; K = 1 ./ (dt + 1);
        seen = R.lastSpk > 0;
        for ii = I
            mj = R.conn(ii,:) & seen & (R.stdpRec(ii,:) ~= R.lastSpk);
            mk = R.conn(:,ii)' & seen & (R.stdpRec(:,ii)' ~= R.lastSpk);
            if doSTDP
                R.W(ii, mj) = R.W(ii, mj) - rate * K(mj);
                R.W(mk, ii) = R.W(mk, ii) + rate * K(mk)';
            end
            if needE
                elig(mk, ii) = elig(mk, ii) + K(mk)';
            end
            R.stdpRec(ii, mj) = R.lastSpk(mj);
            R.stdpRec(mk, ii) = R.lastSpk(mk);
        end
    end
    flag(R.inputIdx) = sp(t, :);
    flag(R.shadow)   = sn(t, :);
    raster(t, :) = flag;
end
rates = mean(raster == 1, 1)';
end

function w = desnn_weights(raster, cfg)
[T, n] = size(raster);
spk = (raster == 1);
nSp = sum(spk, 1);
first = zeros(1, n);
for j = 1:n
    kf = find(spk(:,j), 1, 'first');
    if ~isempty(kf), first(j) = kf; end
end
w = -cfg.drift * T * ones(1, n);
spiked = first > 0;
if any(spiked)
    keyOrd = first(spiked) * (n+1) + find(spiked);
    [~, ordIdx] = sort(keyOrd, 'ascend');
    ranks = zeros(1, nnz(spiked)); ranks(ordIdx) = 1:nnz(spiked);
    later  = T - first(spiked);
    up     = nSp(spiked) - 1;
    down   = later - up;
    w(spiked) = cfg.roMod .^ ranks + cfg.drift * (up - down);
end
end

function th = aer_threshold(Xtr, alpha)
nCh = size(Xtr,2); th = zeros(nCh,1);
for k = 1:size(Xtr,3)
    d = abs(diff(Xtr(:,:,k),1,1));
    th = th + (mean(d,1) + alpha*std(d,0,1))';
end
th = th / size(Xtr,3);  th(th==0) = eps;
end

function [sp, sn] = aer_encode(x, th)
T = size(x,1);
d = x - [x(1,:); x(1:T-1,:)];
thM = repmat(th', T, 1);
exc = d >  thM;  inh = d < -thM;
sp = double(exc) - double(inh);
sn = double(inh);
end

function [sp, sn] = encode_sample(x, th, cfg)
[sp, sn] = aer_encode(x, th);
if isfield(cfg, 'cycleChannels') && cfg.cycleChannels
    s0 = x - mean(x, 1);
    cyc = [zeros(1, size(x,2)); double(diff(sign(s0), 1, 1) ~= 0)];
    sp = [sp, cyc];
    sn = [sn, zeros(size(cyc))];
end
end

function trainIdx = subsample_train(candIdx, y, pid, perClass)
trainIdx = [];
for p = unique(pid(candIdx))'
    for c = unique(y(candIdx))'
        ix = candIdx(pid(candIdx)==p & y(candIdx)==c);
        if numel(ix) > perClass
            ix = ix(round(linspace(1, numel(ix), perClass)));
        end
        trainIdx = [trainIdx; ix(:)]; %#ok<AGROW>
    end
end
end

function yh = wknn_predict(w, W, lab, K, sigma)
d = sqrt(sum((W - w).^2, 2));
[ds, ix] = sort(d, 'ascend');
K = min(K, numel(lab)-1); K = max(K, 1);
Dn = ds(1:K) ./ (ds(K+1) + eps);
Kd = exp(-Dn / (2*sigma^2));
kl = lab(ix(1:K));
cls = unique(lab); score = zeros(numel(cls),1);
for c = 1:numel(cls), score(c) = sum(Kd(kl == cls(c))); end
[~, best] = max(score); yh = cls(best);
end
