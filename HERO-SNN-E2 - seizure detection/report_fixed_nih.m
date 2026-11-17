%% report_fixed_nih.m - Table 2 row for the fixed-parameter E2 run.
%% Reads prepost_final_results_all_full_FX_noHERO.mat and
%% prepost_final_results_all_full_FX.mat from the current folder.
clear; clc;

KEEP = 1:10;
B = load('prepost_final_results_all_full_FX_noHERO.mat'); rb = B.results;
H = load('prepost_final_results_all_full_FX.mat');        rh = H.results;

bv = []; hv = []; aY = []; aB = []; aH = [];
fprintf('%-6s | %7s | %7s | %6s | %s\n', 'fold', 'BASE', 'HERO', 'delta', 'base drift | HERO drift / eta / passes');
for f = KEEP
    b = rb{f}; h = rh{f};
    if isempty(b) || isempty(h), fprintf('%-6d | (not finished)\n', f); continue; end
    bv(end+1) = 100*b.bacc; hv(end+1) = 100*h.bacc; %#ok<*SAGROW>
    aY = [aY; b.ytrue(:)]; aB = [aB; b.pred(:)]; aH = [aH; h.pred(:)];
    fprintf('%-6d | %6.1f%% | %6.1f%% | %+6.1f | %.5g | %.5g / %g / %d\n', f, ...
        bv(end), hv(end), hv(end)-bv(end), b.drift, h.drift, h.eta, h.passes);
end
d = hv - bv; n = numel(d);
fprintf('%-6s | %6.1f%% | %6.1f%% | %+6.1f\n', 'MEAN', mean(bv), mean(hv), mean(d));
fprintf('%-6s | %6.1f  | %6.1f  |\n', 'SD', std(bv), std(hv));
fprintf('folds improved %d/%d, tied %d, worse %d\n\n', nnz(d>0), n, nnz(d==0), nnz(d<0));

for arm = 1:2
    if arm == 1, nm = 'BASELINE'; P = aB; else, nm = 'HERO-SNN'; P = aH; end
    c  = [nnz(aY==1&P==1) nnz(aY==1&P==2); nnz(aY==2&P==1) nnz(aY==2&P==2)];
    r  = [c(1,1)/sum(c(1,:)), c(2,2)/sum(c(2,:))];
    pr = [c(1,1)/max(sum(c(:,1)),1), c(2,2)/max(sum(c(:,2)),1)];
    f1 = 2*pr.*r ./ max(pr + r, eps);
    fprintf('%-8s (n=%d): bAcc %.1f%% | sens (seizure) %.1f%% | spec (non-seizure) %.1f%% | macro-F1 %.3f | conf [%d %d; %d %d]\n', ...
        nm, numel(aY), 100*mean(r), 100*r(2), 100*r(1), mean(f1), c(1,1), c(1,2), c(2,1), c(2,2));
end
rep = nnz(aB ~= aY & aH == aY); brk = nnz(aB == aY & aH ~= aY);
fprintf('\nWilcoxon exact (per-fold bAcc, n=%d): p = %.4g\n', n, wilcoxon_exact(d));
fprintf('McNemar exact: repaired %d, broken %d | p = %.4g\n', rep, brk, mcnemar_p(rep, brk));

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
