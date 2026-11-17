% run_hero_nih10_ablation.m
% Baseline, HERO-SNN and three ablation arms, NIH seizure data, 10 patients,
% leave-one-patient-out, 5 seeds. Parameters are fixed per fold (tuned values).
% Arms: noHERO, HERO, HERO_noHomeo (no threshold adaptation),
% HERO_homeoOnly (eta = 0), HERO_randElig (random eligibility).

% tuned parameters
%        fold  baseDrift  heroDrift  eta     passes   (drift x 0.00256)
FIXED = [  1    0.5        0.5       0.01    2 ;
           2    1          4         0.05    2 ;
           3    1          1         0.01    1 ;
           4    1          2         0.05    1 ;
           5    1          1         0.005   2 ;
           6    1          1         0.01    1 ;
           7    1          2         0.05    2 ;
           8    1          2         0.01    1 ;
           9    4          1         0.05    2 ;
          10    1          4         0.001   1 ];

clear cfg; clc;

cfg.smokeTest   = false;
cfg.dataFile    = 'hero_dataset_nih10_6ch_05s.mat';
cfg.runGroup    = 'all';
cfg.arms        = {'noHERO','HERO','HERO_noHomeo','HERO_homeoOnly','HERO_randElig'};
cfg.lsNorm      = 'matched';
cfg.lsCap       = 5;
cfg.fixIters    = 1;
cfg.eligWindow  = Inf;
cfg.eligLate    = 0;
cfg.epochLen    = Inf;
cfg.HplusIDs    = [1 3 5 8 9 10 11 17 18 19 21 24 25 27];
cfg.snapshotFold = 1;
cfg.reportStats  = true;

cfg.brainCoordsFile = 'brain_coordinates_300.csv';
cfg.eegCoordsFile   = 'eeg_mapping_2pt_12ch.csv';
cfg.cycleChannels   = true;
cfg.inputPlacement  = 'anatomical';
cfg.inputGain       = 2;
cfg.grid            = [6 6 6];

cfg.aerAlpha    = 0.5;
cfg.aerScope    = 'participant';
cfg.swr         = 3.5;
cfg.condVel     = 3.5;
cfg.leak        = 0.002;
cfg.vThresh0    = 0.5;
cfg.refractory  = 6;
cfg.stdpRate    = 0.01;
cfg.posFrac     = 0.8;

cfg.roMod       = 0.8;
cfg.drift       = 0.005 * 512 / 1000;
cfg.K           = 1;
cfg.sigma       = 1;

cfg.eta         = 0.01;
cfg.potRatio    = 0.1;
cfg.mapMode     = 'contrast';
cfg.coreFrac    = 0.8;
cfg.binTopFrac  = 0.15;
cfg.kappa       = 2;
cfg.passes      = 1;
cfg.homeoRate   = 0.01; cfg.homeoTarget = 0.05;
cfg.threshMin   = 0.1;  cfg.threshMax  = 1.5;

cfg.trainPerClass = 40;
cfg.nSeeds        = 5;
cfg.doTuning      = false;
cfg.tuneDrifts    = cfg.drift * [0.5 1 2 4];
cfg.tuneBaseline  = true;
cfg.tuneEtas      = [0.001 0.002 0.005 0.01 0.02 0.05];
cfg.tunePasses    = [1 2];
cfg.nInnerVal     = 4;
cfg.foldsToRun    = FIXED(:,1)';
cfg.FIXED         = FIXED;

if cfg.smokeTest
    cfg.doTuning      = false;
    cfg.nSeeds        = 1;
    cfg.trainPerClass = 10;
    cfg.foldsToRun    = [];
end

S = load(cfg.dataFile);
X = S.X; y = S.y(:); pid = S.pid(:);
assert(ndims(X) == 3, 'X must be time x channels x samples');
if isfield(cfg, 'epochLen') && isfinite(cfg.epochLen) && cfg.epochLen < size(X,1)
    X = X(1:cfg.epochLen, :, :);
    fprintf('[epochLen] epochs truncated to first %d timesteps\n', cfg.epochLen);
end
classes = unique(y(:))';
assert(numel(classes) >= 2 && all(classes == 1:numel(classes)), ...
    'y must contain classes 1..K');
nCh = size(X,2);
nChIn = nCh * (1 + double(cfg.cycleChannels));
fprintf('Dataset: %d time points x %d channels x %d samples, %d classes (chance bAcc = %.1f%%)\n', ...
    size(X,1), nCh, size(X,3), numel(classes), 100/numel(classes));
if cfg.cycleChannels
    fprintf('[encoding] DUAL-STREAM: %d TBR + %d cycle = %d input neurons\n', nCh, nCh, nChIn);
end

allIDs = unique(pid)';
switch cfg.runGroup
    case 'Hplus',  keepIDs = intersect(allIDs, cfg.HplusIDs);
    case 'Hminus', keepIDs = setdiff(allIDs, cfg.HplusIDs);
    case 'all',    keepIDs = allIDs;
end
m = ismember(pid, keepIDs);
X = X(:,:,m); y = y(m); pid = pid(m);
foldIDs = unique(pid)';
fprintf('Group %s: %d patients, %d epochs (%d interictal / %d ictal)\n', ...
    cfg.runGroup, numel(foldIDs), numel(y), nnz(y==1), nnz(y==2));

[coords, chanCoords, coordMode] = load_coords(cfg, nChIn);
fprintf('Reservoir: %d neurons (%s mode) + %d shadow input neurons\n', ...
    size(coords,1), coordMode, nChIn);
if isfinite(cfg.condVel)
    Pall = [coords; chanCoords]; Dtmp = zeros(size(Pall,1));
    for k = 1:3, Dtmp = Dtmp + (Pall(:,k) - Pall(:,k)').^2; end
    Dtmp = sqrt(Dtmp); Dtmp = Dtmp(Dtmp > 0 & Dtmp <= cfg.swr*10);
    dl = max(1, round(Dtmp / cfg.condVel));
    fprintf('[delays] condVel=%.3g mm/step -> connection delays %d..%d steps (%d unique)\n\n', ...
        cfg.condVel, min(dl), max(dl), numel(unique(dl)));
else
    fprintf('[delays] condVel=Inf -> instant transmission (delay-free reference)\n\n');
end

armNames = {'STDP+deSNN baseline', 'HERO-SNN (depression-only)', ...
            'HERO2 (two-sided ablation)', 'HEROMAP (map-modulated depression)', ...
            'HEROLS (readout-aimed depression, e-prop style)'};
for a = 1:numel(cfg.arms)
    armName = cfg.arms{a}; cfg.ablation = '';
    if numel(armName) > 5 && strcmpi(armName(1:5), 'HERO_')
        cfg.ablation = armName(6:end); armName = 'HERO';
    end
    switch upper(armName)
        case 'HERO',    armMode = 1;
        case 'HERO2',   armMode = 2;
        case 'HEROMAP', armMode = 3;
        case 'HEROLS',  armMode = 4;
        otherwise,      armMode = 0;
    end
    fprintf('################ ARM %d/%d: %s %s ################\n', ...
        a, numel(cfg.arms), armNames{armMode+1}, cfg.ablation);
    run_arm(X, y, pid, foldIDs, coords, chanCoords, cfg, armMode);
    fprintf('\n');
end
fprintf('All arms complete. For the network-change figures run plot_hero_changes.m\n');

function run_arm(X, y, pid, foldIDs, coords, chanCoords, cfg, armMode)
heroOn = armMode > 0;
nFolds = numel(foldIDs);
tag = ternary(cfg.smokeTest, 'smoke', 'full');
tag = [tag '_FX'];
if armMode > 0 && ~isempty(cfg.ablation), tag = [tag '_AB_' cfg.ablation]; end
if strcmpi(cfg.inputPlacement, 'spread'), tag = ['spread_' tag]; end
if armMode == 0, tag = [tag '_noHERO']; end
if armMode == 2, tag = [tag '_HERO2'];  end
if armMode == 3, tag = [tag '_HEROMAP']; end
if armMode == 4, tag = [tag '_HEROLS']; end
if cfg.fixIters > 1 && armMode > 0, tag = [tag '_FIX']; end
if isfinite(cfg.eligWindow) && armMode > 0, tag = [tag '_EW']; end
if isfield(cfg, 'epochLen') && isfinite(cfg.epochLen), tag = [tag sprintf('_T%d', cfg.epochLen)]; end
diary(sprintf('prepost_final_%s_%s_log.txt', cfg.runGroup, tag));
armLbl = 'baseline'; if armMode==1, armLbl='HERO'; elseif armMode==2, armLbl='HERO2'; elseif armMode==3, armLbl='HEROMAP'; elseif armMode==4, armLbl='HEROLS'; end
fprintf('=== arm=%s | group=%s | mode=%s | %s ===\n', armLbl, cfg.runGroup, tag, datestr(now));

foldsToRun = cfg.foldsToRun; if isempty(foldsToRun), foldsToRun = 1:nFolds; end

ckptFile = sprintf('ckpt_final_%s_%s.mat', cfg.runGroup, tag);
if isfile(ckptFile)
    C = load(ckptFile); results = C.results; doneFolds = C.doneFolds;
    fprintf('Resuming: %d folds already done in %s\n', nnz(doneFolds), ckptFile);
else
    results = cell(1,nFolds); doneFolds = false(1,nFolds);
end

for f = foldsToRun
    if doneFolds(f), fprintf('Fold %d (P%d): already done, skipping\n', f, foldIDs(f)); continue; end
    tFold = tic; testID = foldIDs(f);
    fprintf('\n--- Fold %d/%d | test participant P%d ---\n', f, nFolds, testID);

    isTest   = (pid == testID);
    trainIdx = subsample_train(find(~isTest), y, pid, cfg.trainPerClass);
    testIdx  = find(isTest);

    SPp = cell(size(X,3),1); SPn = SPp;
    useIdx = [trainIdx(:); testIdx(:)];
    aerScope = cfg.aerScope;
    if strcmpi(aerScope, 'participant') && numel(unique(pid)) == numel(pid)
        aerScope = 'train';
        fprintf('   [encode] pids are one-per-sample: participant scope would self-normalize each sample; using train scope\n');
    end
    if strcmpi(aerScope, 'participant')
        for p = unique(pid(useIdx))'
            pix = useIdx(pid(useIdx) == p);
            thp = aer_threshold(X(:,:,pix), cfg.aerAlpha);
            for n = pix'
                [SPp{n}, SPn{n}] = encode_sample(X(:,:,n), thp, cfg);
            end
        end
    else
        th = aer_threshold(X(:,:,trainIdx), cfg.aerAlpha);
        for n = useIdx'
            [SPp{n}, SPn{n}] = encode_sample(X(:,:,n), th, cfg);
        end
    end
    dens = 0; nlow = 0; nhigh = 0; nc = size(X,2);
    for n = useIdx'
        d = mean(SPp{n} ~= 0, 1); dens = dens + mean(d);
        nlow = nlow + nnz(d < 0.005); nhigh = nhigh + nnz(d > 0.4);
    end
    dens = dens / numel(useIdx);
    fprintf('   [encode:%s] mean spike density %.3f; near-silent ch %.1f/epoch, saturated ch %.1f/epoch\n', ...
        aerScope, dens, nlow/numel(useIdx), nhigh/numel(useIdx));
    if dens < 0.01 || dens > 0.5
        fprintf('   [encode] WARNING: spike density outside healthy range - consider adjusting cfg.aerAlpha\n');
    end

    global TUNEPHASE
    if cfg.doTuning && (armMode > 0 || cfg.tuneBaseline)
        TUNEPHASE = 1;
        [bestEta, bestPasses, bestDrift] = tune_fold(SPp, SPn, y, pid, trainIdx, ...
                                                     coords, chanCoords, cfg, f, armMode);
        TUNEPHASE = 0;
    else
        TUNEPHASE = 0;
        k = find(cfg.FIXED(:,1) == f, 1);
        assert(~isempty(k), 'fold %d has no row in the FIXED table', f);
        if heroOn
            bestDrift = 0.00256 * cfg.FIXED(k,3); bestEta = cfg.FIXED(k,4); bestPasses = cfg.FIXED(k,5);
            if strcmpi(cfg.ablation, 'homeoOnly'), bestEta = 0; end
        else
            bestDrift = 0.00256 * cfg.FIXED(k,2); bestEta = 0; bestPasses = 0;
        end
    end
    cfgRun = cfg; cfgRun.drift = bestDrift;
    fprintf('   using drift=%.3g, eta=%.3g, passes=%d\n', bestDrift, bestEta, bestPasses);

    predSeeds = zeros(numel(testIdx), cfg.nSeeds);
    stats = []; 
    for s = 1:cfg.nSeeds
        rng(1000*f + s, 'twister');
        wantSnap  = heroOn && (s == 1) && bestPasses > 0;
        wantStats = wantSnap || (heroOn && (s == 1) && bestPasses > 0 && cfg.reportStats);
        [predSeeds(:,s), st, snap] = train_and_predict(SPp, SPn, y, trainIdx, testIdx, ...
                              bestEta, bestPasses, coords, chanCoords, cfgRun, wantStats, wantSnap, armMode);
        if wantStats, stats = st; end
        if wantSnap && ~isempty(snap)
            snapFile = sprintf('hero_snapshot_%s_%s_f%d.mat', cfg.runGroup, tag, f);
            save(snapFile, '-struct', 'snap', '-v7.3');
            fprintf('   [snapshot] before/after cube saved to %s\n', snapFile);
        end
    end
    predWin = mode_with_tiebreak(predSeeds);

    r.testID = testID; r.testIdx = testIdx; r.ytrue = y(testIdx);
    r.pred = predWin; r.predSeeds = predSeeds; r.stats = stats;
    r.eta = bestEta; r.passes = bestPasses; r.drift = bestDrift;
    r.runtimeSec = toc(tFold);
    [r.bacc, r.rec, r.conf] = fold_metrics(r.ytrue, r.pred);
    results{f} = r; doneFolds(f) = true;
    save(ckptFile, 'results', 'doneFolds', 'cfg', '-v7.3');
    rstr = sprintf('%.0f%% ', 100*r.rec(~isnan(r.rec)));
    fprintf('   P%d: bAcc=%.1f%%  per-class recall [%s] [%.1f min, saved]\n', ...
        testID, 100*r.bacc, strtrim(rstr), r.runtimeSec/60);
    if ~isempty(stats) && cfg.reportStats
        fprintf(['   HERO changes: %d synapses depressed (%.1f%% of positive internal), ' ...
                 'mean dW=-%.4f, %d driven to 0\n'], ...
            stats.nDepressed, 100*stats.fracDepressed, stats.meanDep, stats.nZeroed);
        fprintf(['   homeostasis: thresholds up %d / down %d of %d internal neurons ' ...
                 '(mean |dTh|=%.4f); mean internal rate %.4f -> %.4f\n'], ...
            stats.nThreshUp, stats.nThreshDown, stats.nInternal, ...
            stats.meanAbsDTh, stats.ratePre, stats.ratePost);
        fprintf('   error signal fired: [%s] of %d training samples per pass\n', ...
            strtrim(sprintf('%d ', stats.heroFires)), numel(trainIdx));
    end
end

done = find(doneFolds);
if isempty(done), fprintf('No folds completed.\n'); diary off; return; end
fprintf('\n===== SUMMARY | arm=%s (%d/%d folds) =====\n', armLbl, numel(done), nFolds);
fprintf('%6s %8s   %s\n','PID','bAcc','per-class recall');
B = zeros(numel(done),1); allY = []; allP = [];
for i = 1:numel(done)
    r = results{done(i)}; B(i) = r.bacc;
    allY = [allY; r.ytrue]; allP = [allP; r.pred];
    rstr = sprintf('%.0f%% ', 100*r.rec(~isnan(r.rec)));
    fprintf('%6d %7.1f%%   [%s]\n', r.testID, 100*r.bacc, strtrim(rstr));
end
[gb, grec, gconf] = fold_metrics(allY, allP);
fprintf('Mean per-fold bAcc: %.1f%%\n', 100*mean(B));
rstr = sprintf('%.1f%% ', 100*grec(~isnan(grec)));
fprintf('Pooled: bAcc=%.1f%%  per-class recall [%s]  overallAcc=%.1f%%\n', ...
    100*gb, strtrim(rstr), 100*mean(allY==allP));
fprintf('Pooled confusion (rows=true, cols=predicted):\n');
disp(gconf);
resultsFile = sprintf('prepost_final_results_%s_%s.mat', cfg.runGroup, tag);
save(resultsFile, 'results', 'cfg', '-v7.3');
fprintf('Results saved to %s\n', resultsFile);
diary off;
end

function [coords, chanCoords, mode] = load_coords(cfg, nCh)
useTemplate = false;
if isfile(cfg.brainCoordsFile) && isfile(cfg.eegCoordsFile)
    coords = readmatrix_compat(cfg.brainCoordsFile);
    chanCoords = readmatrix_compat(cfg.eegCoordsFile);
    if size(chanCoords,1) == nCh
        useTemplate = true;
    else
        fprintf(['NOTE: eeg mapping has %d rows but data has %d channels - ' ...
                 'falling back to grid mode.\n'], size(chanCoords,1), nCh);
    end
end
if useTemplate
    mode = 'brain-template';
    if isfield(cfg, 'inputPlacement') && strcmpi(cfg.inputPlacement, 'spread')
        dual = isfield(cfg,'cycleChannels') && cfg.cycleChannels && mod(nCh,2)==0;
        if dual, nSites = nCh/2; else, nSites = nCh; end
        ctr = mean(coords, 1);
        [~, s1] = max(sum((coords - ctr).^2, 2));
        sel = s1;
        dmin = sqrt(sum((coords - coords(s1,:)).^2, 2));
        for k = 2:nSites
            [~, nxt] = max(dmin);
            sel(end+1) = nxt;
            dmin = min(dmin, sqrt(sum((coords - coords(nxt,:)).^2, 2)));
        end
        sites = coords(sel, :);
        if dual, chanCoords = [sites; sites + repmat([0 0 8], nSites, 1)];
        else, chanCoords = sites; end
        mode = 'brain-template SPREAD-inputs';
        fprintf('[placement] %d channel sites spread by farthest-point sampling (min site distance %.1f mm)\n', ...
            nSites, min(pdist_min(sites)));
    end
else
    [gx, gy, gz] = ndgrid(1:cfg.grid(1), 1:cfg.grid(2), 1:cfg.grid(3));
    coords = 10 * [gx(:) gy(:) gz(:)];
    pick = round(linspace(1, size(coords,1), nCh));
    chanCoords = coords(pick, :);
    mode = 'grid-fallback';
end
end

function d = pdist_min(P)
m = size(P,1); d = inf;
for i = 1:m-1
    di = sqrt(sum((P(i+1:end,:) - P(i,:)).^2, 2));
    d = min(d, min(di));
end
end

function M = readmatrix_compat(fn)
try, M = readmatrix(fn); catch, M = csvread(fn); end
M = M(all(isfinite(M),2), :);
end

function th = aer_threshold(Xtr, alpha)
nCh = size(Xtr,2); th = zeros(nCh,1);
for k = 1:size(Xtr,3)
    d = abs(diff(Xtr(:,:,k),1,1));
    th = th + (mean(d,1) + alpha*std(d,0,1))';
end
th = th / size(Xtr,3);  th(th==0) = eps;
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

function [sp, sn] = aer_encode(x, th)
T = size(x,1);
d = x - [x(1,:); x(1:T-1,:)];
thM = repmat(th', T, 1);
exc = d >  thM;  inh = d < -thM;
sp = double(exc) - double(inh);
sn = double(inh);
end

function trainIdx = subsample_train(candIdx, y, pid, perClass)
trainIdx = [];
for p = unique(pid(candIdx))'
    for c = unique(y(candIdx))'
        ix = candIdx(pid(candIdx)==p & y(candIdx)==c);
        if numel(ix) > perClass
            ix = ix(round(linspace(1, numel(ix), perClass)));
        end
        trainIdx = [trainIdx; ix(:)];
    end
end
end

function [bestEta, bestPasses, bestDrift] = tune_fold(SPp, SPn, y, pid, trainIdx, coords, chanCoords, cfg, f, armMode)
rng(500 + f, 'twister');
tp = unique(pid(trainIdx))';
if numel(tp) >= 2
    nv = min(cfg.nInnerVal, max(1, numel(tp)-2));
    valP = tp(randperm(numel(tp), nv));
    inTr = trainIdx(~ismember(pid(trainIdx), valP));
    inVa = trainIdx( ismember(pid(trainIdx), valP));
else
    inTr = []; inVa = [];
    for c = unique(y(trainIdx))'
        ci = trainIdx(y(trainIdx) == c);
        ci = ci(randperm(numel(ci)));
        k  = max(1, round(0.25 * numel(ci)));
        inVa = [inVa; ci(1:k)]; inTr = [inTr; ci(k+1:end)];
    end
    inTr = sort(inTr); inVa = sort(inVa);
    fprintf('   [tune] single-participant training set: stratified inner split %d train / %d val\n', ...
        numel(inTr), numel(inVa));
end
best1 = -inf; bestDrift = cfg.tuneDrifts(1);
for dv = cfg.tuneDrifts
    cfgT = cfg; cfgT.drift = dv;
    rng(7000 + 13*f, 'twister');
    pr = train_and_predict(SPp, SPn, y, inTr, inVa, 0, 0, coords, chanCoords, cfgT, false, false, 0);
    b = fold_metrics(y(inVa), pr);
    fprintf('   [tune S1 baseline] drift=%.3g -> inner bAcc %.1f%%\n', dv, 100*b);
    if b > best1, best1 = b; bestDrift = dv; end
end
fprintf('   [tune S1] selected drift=%.3g (inner bAcc %.1f%%)\n', bestDrift, 100*best1);
bestEta = 0; bestPasses = 0;
if armMode == 0, return; end
best2 = -inf; bestEta = cfg.tuneEtas(1); bestPasses = cfg.tunePasses(1);
cfgT = cfg; cfgT.drift = bestDrift;
for e = cfg.tuneEtas
    for p = cfg.tunePasses
        rng(7000 + 13*f, 'twister');
        pr = train_and_predict(SPp, SPn, y, inTr, inVa, e, p, coords, chanCoords, cfgT, false, false, armMode);
        b = fold_metrics(y(inVa), pr);
        fprintf('   [tune S2 HERO] drift=%.3g eta=%.3g passes=%d -> inner bAcc %.1f%%\n', bestDrift, e, p, 100*b);
        if b > best2, best2 = b; bestEta = e; bestPasses = p; end
    end
end
fprintf('   [tune S2] selected eta=%.3g passes=%d (inner bAcc %.1f%% vs stage-1 baseline %.1f%%)\n', ...
    bestEta, bestPasses, 100*best2, 100*best1);
end

function [pred, stats, snap] = train_and_predict(SPp, SPn, y, trainIdx, testIdx, ...
                                eta, passes, coords, chanCoords, cfg, wantStats, wantSnap, armMode)
stats = []; snap = [];
R = build_cube(coords, chanCoords, cfg);

ord = trainIdx(randperm(numel(trainIdx)));
for n = ord(:)'
    [~, R] = propagate(SPp{n}, SPn{n}, R, cfg, true, false);
end

W = zeros(numel(trainIdx), R.n);
ratePre = zeros(R.n, 1);
for i = 1:numel(trainIdx)
    [raster, R, ~, rts] = propagate(SPp{trainIdx(i)}, SPn{trainIdx(i)}, R, cfg, false, false);
    W(i,:) = desnn_weights(raster, cfg);
    ratePre = ratePre + rts;
end
ratePre = ratePre / numel(trainIdx);
lab = y(trainIdx);

global TUNEPHASE
if isempty(TUNEPHASE) || ~TUNEPHASE
    nerr = 0;
    for i = 1:numel(lab)
        others = true(numel(lab),1); others(i) = false;
        if wknn_predict(W(i,:), W(others,:), lab(others), cfg.K, cfg.sigma) ~= lab(i)
            nerr = nerr + 1;
        end
    end
    fprintf('   [fuel] training samples misclassified in-sample: %d of %d\n', nerr, numel(lab));
end

if wantStats
    W0 = R.W; th0 = R.thresh;
end
heroFires = zeros(1, max(passes,1));
wCap = max([R.W(R.plastic & (R.W > 0)); 1]);
nCls = max(lab);
mapO = cell(1, nCls); mapN = zeros(1, nCls);
for cM = 1:nCls, mapO{cM} = zeros(R.n); end
znProt = 0; znAmp = 0; znElig = 0; znEvents = 0;
lsEligN = 0; lsProtN = 0; lsLsum = 0; lsEvents = 0;
fixN = 0; fixHit = 0; fixIterSum = 0;

ratePost = ratePre;
for pass = 1:passes
    for i = 1:numel(trainIdx)
        [raster, R, elig, rates] = propagate(SPp{trainIdx(i)}, SPn{trainIdx(i)}, R, cfg, false, true);
        if max(elig(:)) > 0, elig = elig / max(elig(:)); end
        wi = desnn_weights(raster, cfg);
        others = true(numel(lab),1); others(i) = false;
        yh = wknn_predict(wi, W(others,:), lab(others), cfg.K, cfg.sigma);
        if yh ~= lab(i)
            if wantStats, heroFires(pass) = heroFires(pass) + 1; end
            nIt = max(round(cfg.fixIters), 1); itUsed = nIt; wasFixed = false;
            for itFix = 1:nIt
            mask = R.plastic & (R.W > 0);
            coreT = []; coreP = [];
            if armMode == 3 && mapN(lab(i)) >= 3
                coreT = mapO{lab(i)} >= cfg.coreFrac * mapN(lab(i));
            end
            if armMode == 3 && yh >= 1 && yh <= nCls && mapN(yh) >= 3
                coreP = mapO{yh} >= cfg.coreFrac * mapN(yh);
            end
            if armMode == 3 && ~isempty(coreT)
                G = ones(R.n);
                if strcmpi(cfg.mapMode, 'contrast') && ~isempty(coreP)
                    G(coreP & ~coreT) = cfg.kappa;
                end
                G(coreT) = 0;
                R.W(mask) = max(R.W(mask) - eta*(G(mask).*elig(mask)), 0);
                eOn = mask & (elig > 0);
                znElig = znElig + nnz(eOn); znEvents = znEvents + 1;
                znProt = znProt + nnz(eOn & coreT);
                if ~isempty(coreP), znAmp = znAmp + nnz(eOn & coreP & ~coreT); end
            elseif armMode == 4
                muT = mean(W(others & (lab == lab(i)), :), 1);
                muP = mean(W(others & (lab == yh),     :), 1);
                L = max(muP - muT, 0)';
                if any(~isfinite(L)), L = ones(R.n,1); end
                if max(L) > 0, L = L / max(L); end
                eOn = mask & (elig > 0);
                [~, jj] = find(eOn);
                if strcmpi(cfg.lsNorm, 'matched') && ~isempty(jj)
                    mL = mean(L(jj));
                    if mL > 0, L = min(L / mL, cfg.lsCap); end
                end
                eligL = bsxfun(@times, elig, L');
                R.W(mask) = max(R.W(mask) - eta*eligL(mask), 0);
                lsEvents = lsEvents + 1; lsEligN = lsEligN + nnz(eOn);
                lsProtN = lsProtN + nnz(eOn(:, L == 0));
                lsLsum = lsLsum + sum(L(jj));
            else
                if strcmpi(cfg.ablation, 'randElig')
                    v = elig(mask); v = v(randperm(numel(v)));
                    R.W(mask) = max(R.W(mask) - eta*v, 0);
                else
                    R.W(mask) = max(R.W(mask) - eta*elig(mask), 0);
                end
            end
            if itFix == nIt, break; end
            [raster, R, elig, rates] = propagate(SPp{trainIdx(i)}, SPn{trainIdx(i)}, R, cfg, false, true);
            if max(elig(:)) > 0, elig = elig / max(elig(:)); end
            yh = wknn_predict(desnn_weights(raster, cfg), W(others,:), lab(others), cfg.K, cfg.sigma);
            if yh == lab(i), wasFixed = true; itUsed = itFix; break; end
            end
            if nIt > 1
                fixN = fixN + 1; fixIterSum = fixIterSum + itUsed;
                fixHit = fixHit + double(wasFixed);
            end
        elseif armMode == 2
            mask = R.plastic & (R.W > 0);
            R.W(mask) = min(R.W(mask) + cfg.potRatio*eta*elig(mask), wCap);
        elseif armMode == 3
            pe = elig(elig > 0);
            if ~isempty(pe)
                ps = sort(pe, 'descend');
                thB = ps(max(1, ceil(cfg.binTopFrac * numel(ps))));
                mapO{lab(i)} = mapO{lab(i)} + double(elig >= thB);
                mapN(lab(i)) = mapN(lab(i)) + 1;
            end
        end
        if ~strcmpi(cfg.ablation, 'noHomeo')
            adj = cfg.homeoRate * (rates - cfg.homeoTarget);
            adj(R.inputIdx) = 0; adj(R.shadow) = 0;
            R.thresh = min(max(R.thresh + adj, cfg.threshMin), cfg.threshMax);
        end
    end
    ratePost = zeros(R.n, 1);
    for i = 1:numel(trainIdx)
        [raster, R, ~, rts] = propagate(SPp{trainIdx(i)}, SPn{trainIdx(i)}, R, cfg, false, false);
        W(i,:) = desnn_weights(raster, cfg);
        ratePost = ratePost + rts;
    end
    ratePost = ratePost / numel(trainIdx);
end

if armMode == 3 && znEvents > 0 && (isempty(TUNEPHASE) || ~TUNEPHASE)
    fprintf('   [zones] per error: protected %.1f%%, amplified %.1f%% of eligible (over %d map-gated errors)\n', ...
        100*znProt/max(znElig,1), 100*znAmp/max(znElig,1), znEvents);
end
if armMode == 4 && lsEvents > 0 && (isempty(TUNEPHASE) || ~TUNEPHASE)
    fprintf('   [LS] %d aimed updates: %.1f%% of eligible synapses protected (target L=0), mean L on the rest %.2f\n', ...
        lsEvents, 100*lsProtN/max(lsEligN,1), lsLsum/max(lsEligN-lsProtN,1));
end
if fixN > 0 && (isempty(TUNEPHASE) || ~TUNEPHASE)
    fprintf('   [fix] %d errors: %.0f%% fixed within %d updates (mean %.1f updates per error)\n', ...
        fixN, 100*fixHit/fixN, round(cfg.fixIters), fixIterSum/fixN);
end
if wantStats
    stats = change_stats(W0, R.W, th0, R.thresh, R, ratePre, ratePost, heroFires);
    if wantSnap
        snap.W0 = W0; snap.W1 = R.W; snap.th0 = th0; snap.th1 = R.thresh;
        snap.conn = R.conn; snap.plastic = R.plastic;
        snap.coords = R.P; snap.inputIdx = R.inputIdx; snap.shadow = R.shadow;
        snap.ratePre = ratePre; snap.ratePost = ratePost;
        snap.heroFires = heroFires; snap.eta = eta; snap.passes = passes;
        if armMode == 3
            for cM = 1:nCls
                snap.(sprintf('freqC%d', cM)) = mapO{cM} / max(mapN(cM), 1);
                snap.(sprintf('coreC%d', cM)) = mapO{cM} >= cfg.coreFrac * max(mapN(cM), 1);
            end
        end
    end
end

pred = zeros(numel(testIdx),1);
for i = 1:numel(testIdx)
    [raster, R] = propagate(SPp{testIdx(i)}, SPn{testIdx(i)}, R, cfg, false, false);
    pred(i) = wknn_predict(desnn_weights(raster, cfg), W, lab, cfg.K, cfg.sigma);
end
end

function st = change_stats(W0, W1, th0, th1, R, ratePre, ratePost, heroFires)
internal = true(R.n,1); internal(R.inputIdx) = false; internal(R.shadow) = false;
posMask = R.plastic & (W0 > 0);
dW = W0 - W1;
chg = posMask & (dW > 1e-12);
st.nDepressed    = nnz(chg);
st.fracDepressed = st.nDepressed / max(nnz(posMask),1);
st.meanDep       = mean(dW(chg));           if isnan(st.meanDep), st.meanDep = 0; end
st.maxDep        = max([dW(chg); 0]);
st.nZeroed       = nnz(posMask & (W1 == 0));
dTh = th1 - th0;
st.nInternal   = nnz(internal);
st.nThreshUp   = nnz(internal & (dTh >  1e-12));
st.nThreshDown = nnz(internal & (dTh < -1e-12));
st.meanAbsDTh  = mean(abs(dTh(internal)));
st.ratePre  = mean(ratePre(internal));
st.ratePost = mean(ratePost(internal));
st.heroFires = heroFires;
end

function R = build_cube(coords, chanCoords, cfg)
nRes = size(coords,1); nCh = size(chanCoords,1);
inputIdx = zeros(nCh,1);
taken = false(nRes,1);
for c = 1:nCh
    d2 = sum((coords - chanCoords(c,:)).^2, 2);
    d2(taken) = inf;
    [~, inputIdx(c)] = min(d2); taken(inputIdx(c)) = true;
end
P = [coords; chanCoords];
n = nRes + nCh; shadow = (nRes+1):n;
D = zeros(n);
for k = 1:3, D = D + (P(:,k) - P(:,k)').^2; end
D = sqrt(D);
Dinv = 1./D; Dinv(D==0) = 0;
Wm = sign(rand(n) - (1-cfg.posFrac)) .* rand(n) .* Dinv;
conn = true(n); conn(1:n+1:end) = false;
conn(D > cfg.swr*10) = false;
conn(:, shadow) = false;
conn(:, inputIdx) = false;
both = conn & conn';
[bi, bj] = find(triu(both,1));
for k = 1:numel(bi)
    if rand < 0.5, conn(bi(k), bj(k)) = false; else, conn(bj(k), bi(k)) = false; end
end
W = Wm .* conn;
W(shadow, :) = 0;
if isfield(cfg, 'inputGain'), g = cfg.inputGain; else, g = 2; end
W(inputIdx, :) = g*abs(W(inputIdx, :));
R.n = n; R.W = W; R.conn = conn; R.P = P;
R.inputIdx = inputIdx; R.shadow = shadow;
R.plastic = conn; R.plastic(inputIdx, :) = false;
R.thresh = cfg.vThresh0 * ones(n,1);
R.stdpRec = zeros(n); R.lastSpk = zeros(1,n); R.tGlobal = 0;
if isfinite(cfg.condVel), Dly = max(1, round(D / cfg.condVel)); else, Dly = ones(n); end
Dly(~conn) = 0;
R.dlyVals = unique(Dly(conn))';
R.maxDly  = max([R.dlyVals, 1]);
R.DlyMask = cell(1, numel(R.dlyVals));
for k = 1:numel(R.dlyVals), R.DlyMask{k} = (Dly == R.dlyVals(k)); end
end

function [raster, R, elig, rates] = propagate(sp, sn, R, cfg, doSTDP, doElig)
T = size(sp,1); n = R.n;
flag = zeros(1,n); pot = zeros(1,n); refr = zeros(1,n);
nB = R.maxDly + 1; B = zeros(nB, n);
raster = zeros(T, n);
needE = (nargout >= 3) && doElig;
if needE, elig = zeros(n); else, elig = []; end
rate = cfg.stdpRate * 0.1;
if isfield(cfg, 'eligWindow') && isfinite(cfg.eligWindow), eW = cfg.eligWindow; else, eW = Inf; end
if isfield(cfg, 'eligLate'), eL = cfg.eligLate; else, eL = 0; end
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
                if t <= eW
                    elig(mk, ii) = elig(mk, ii) + K(mk)';
                elseif eL > 0
                    elig(mk, ii) = elig(mk, ii) + eL * K(mk)';
                end
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
    k = find(spk(:,j), 1, 'first');
    if ~isempty(k), first(j) = k; end
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

function m = mode_with_tiebreak(P)
m = zeros(size(P,1),1);
for i = 1:size(P,1)
    u = unique(P(i,:)); cnt = zeros(size(u));
    for k = 1:numel(u), cnt(k) = nnz(P(i,:)==u(k)); end
    [~, b] = max(cnt); m(i) = u(b);
end
end

function [bacc, rec, conf] = fold_metrics(ytrue, pred)
K = max([ytrue(:); pred(:)]);
conf = zeros(K);
for i = 1:numel(ytrue), conf(ytrue(i), pred(i)) = conf(ytrue(i), pred(i)) + 1; end
present = unique(ytrue(:))';
rec = nan(1, K);
for c = present, rec(c) = conf(c,c) / max(sum(conf(c,:)), 1); end
bacc = mean(rec(present));
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
