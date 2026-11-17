% report_ablation_nih.m - ablation table for the NIH seizure run (fixed parameters).
% Reads the _FX result files from the current folder; arms not yet run are skipped.
clear; clc;

KEEP   = 1:10;
names  = {'baseline', 'HERO', 'noHomeo', 'homeoOnly', 'randElig'};
files  = {'prepost_final_results_all_full_FX_noHERO.mat', ...
          'prepost_final_results_all_full_FX.mat', ...
          'prepost_final_results_all_full_FX_AB_noHomeo.mat', ...
          'prepost_final_results_all_full_FX_AB_homeoOnly.mat', ...
          'prepost_final_results_all_full_FX_AB_randElig.mat'};
labels = {'STDP+deSNN baseline', 'HERO-SNN (full)', 'HERO without homeostasis', ...
          'Homeostasis only (eta = 0)', 'HERO with shuffled eligibility'};

R = cell(1, numel(names));
for c = 1:numel(names)
    if isfile(files{c}), S = load(files{c}); R{c} = S.results; end
end
iH = 2;
assert(~isempty(R{1}) && ~isempty(R{iH}), 'baseline and HERO result files are required');

B = nan(numel(names), numel(KEEP)); Y = []; P = cell(1, numel(names));
for c = 1:numel(names)
    if isempty(R{c}), continue; end
    yy = []; pp = [];
    for k = 1:numel(KEEP)
        r = R{c}{KEEP(k)};
        if isempty(r), continue; end
        B(c,k) = 100 * r.bacc;
        yy = [yy; r.ytrue(:)]; pp = [pp; r.pred(:)]; %#ok<AGROW>
    end
    P{c} = pp; if c == 1, Y = yy; end
end

fprintf('===== ABLATION | NIH seizure, %d patients, fixed parameters =====\n\n', numel(KEEP));
fprintf('%-32s | %14s | %6s | %6s | %6s | %6s | %7s | %9s | %9s | %9s\n', ...
    'Arm', 'bAcc mean+-SD', 'pooled', 'sens', 'spec', 'mF1', 'improv', 'Wx vs B', 'Wx vs H', 'McN vs B');
for c = 1:numel(names)
    if isempty(R{c}) || isempty(P{c}) || numel(P{c}) ~= numel(Y)
        fprintf('%-32s | (not yet run)\n', labels{c}); continue;
    end
    [pb, se, sp, f1] = pooled(Y, P{c});
    if c == 1
        fprintf('%-32s | %5.1f +- %4.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %.3f | %7s | %9s | %9s | %9s\n', ...
            labels{c}, mean(B(c,:)), std(B(c,:)), pb, se, sp, f1, '-', '-', '-', '-');
        continue;
    end
    d  = B(c,:) - B(1,:);
    pB = wilcoxon_exact(d);
    if c == iH, pHs = '-'; else, pHs = sprintf('%.3g', wilcoxon_exact(B(c,:) - B(iH,:))); end
    rep = nnz(P{1} ~= Y & P{c} == Y); brk = nnz(P{1} == Y & P{c} ~= Y);
    fprintf('%-32s | %5.1f +- %4.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %.3f | %4d/%-2d | %9.3g | %9s | %9.3g\n', ...
        labels{c}, mean(B(c,:)), std(B(c,:)), pb, se, sp, f1, nnz(d > 0), numel(d), pB, pHs, mcnemar_p(rep, brk));
end
fprintf('\nWx vs B / Wx vs H: exact Wilcoxon on per-fold bAcc vs baseline / vs full HERO. McN vs B: exact McNemar on paired epochs vs baseline.\n');
fprintf('\nPer-fold bAcc (columns = folds %s):\n', mat2str(KEEP));
for c = 1:numel(names)
    if all(isnan(B(c,:))), continue; end
    fprintf('%-32s %s\n', labels{c}, sprintf('%6.1f', B(c,:)));
end

function [bq, se, sp, mf1] = pooled(yq, pq)
cmq = [nnz(yq==1&pq==1) nnz(yq==1&pq==2); nnz(yq==2&pq==1) nnz(yq==2&pq==2)];
rq  = [cmq(1,1)/sum(cmq(1,:)), cmq(2,2)/sum(cmq(2,:))];
prq = [cmq(1,1)/max(sum(cmq(:,1)),1), cmq(2,2)/max(sum(cmq(:,2)),1)];
f1q = 2*prq.*rq ./ max(prq + rq, eps);
bq = 100*mean(rq); se = 100*rq(2); sp = 100*rq(1); mf1 = mean(f1q);
end

function pw = wilcoxon_exact(dw)
dw = dw(:); dw = dw(abs(dw) > 1e-9); nw = numel(dw);
if nw == 0, pw = 1; return; end
[~, o] = sort(abs(dw)); rk = zeros(nw,1); rk(o) = 1:nw; aw = abs(dw);
for v = unique(aw)', m = abs(aw - v) < 1e-9; rk(m) = mean(rk(m)); end
Wobs = sum(rk(dw > 0)); mu = sum(rk)/2; cnt = 0; tot = 2^nw;
for k = 0:tot-1
    s = bitget(k, 1:nw)';
    if abs(sum(rk(s == 1)) - mu) >= abs(Wobs - mu) - 1e-9, cnt = cnt + 1; end
end
pw = cnt / tot;
end

function pm = mcnemar_p(nb, nc)
nn = nb + nc; if nn == 0, pm = 1; return; end
km = min(nb, nc); lp = @(j) gammaln(nn+1) - gammaln(j+1) - gammaln(nn-j+1) - nn*log(2);
tm = 0; for j = 0:km, tm = tm + exp(lp(j)); end
pm = min(1, 2*tm);
end
