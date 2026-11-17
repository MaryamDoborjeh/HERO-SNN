E2 - seizure detection, NIH intracranial EEG
============================================

This folder has the code, data and results for experiment E2 of the HERO-SNN
paper (Neuroinformatics). The task is seizure vs non-seizure classification in
10 patients from the Epilepsy iEEG Multicenter Dataset (OpenNeuro ds003029,
v1.0.7), 6 subtemporal channels (AST1-AST4, PST1-PST2), 0.5-s epochs,
leave-one-subject-out. Baseline is STDP + deSNN, HERO-SNN is the same pipeline
plus the refinement stage. The folder also has the three ablation runs of
Table 2 (no homeostasis, no error-driven update, random eligibility).

What is here
------------
run_hero_nih10_ablation.m
    main script. Runs five arms for all 10 folds, 5 seeds per fold, majority
    vote: noHERO (baseline), HERO, HERO_noHomeo, HERO_homeoOnly (eta = 0),
    HERO_randElig. The per-fold drift / eta / passes (Table 3b) are in the
    FIXED table at the top of the file. The first two arms give Table 1, the
    other three give Table 2.
report_fixed_nih.m
    reads the baseline and HERO-SNN result files and prints the per-patient
    table, pooled confusion matrices, Wilcoxon and McNemar (Table 1, Table A2).
report_ablation_nih.m
    reads all five result files and prints Table 2.
plot_snn_connections_FX.m, plot_hero_weights_pool_FX.m, plot_hero_consensus_E2_FX.m
    make the E2 panels of Fig. 4 and Fig. 5 from the snapshot files.
brain_coordinates_300.csv, eeg_mapping_2pt_12ch.csv
    the 300-neuron Talairach template and the input placement for the 6
    channels (each channel feeds a positive and a negative input neuron).
hero_dataset_nih10_6ch_05s.mat
    the sampled dataset used in the paper: X (500 x 6 x 2640, time x channels x
    epochs), y (1 = non-seizure, 2 = seizure), pid (patient 1-10). Built from
    ds003029 as described in Section 3.1 of the paper (60 epochs per class per
    seizure, non-seizure from the 30 s before onset, seizure between onset and
    offset). Patient order: pt01, pt2, pt3, pt8, pt10, pt12, pt14, pt15, pt16, pt17.
prepost_final_results_all_full_FX_noHERO.mat, ..._FX.mat,
..._FX_AB_noHomeo.mat, ..._FX_AB_homeoOnly.mat, ..._FX_AB_randElig.mat
    results of the five arms (per fold and per seed predictions plus the
    parameters used). All numbers in Tables 1, 2 and A2 come from these.
prepost_final_all_full_FX*_log
    console output of the five runs.
hero_snapshot_all_full_FX_f1 ... f10.mat (and the _AB_ versions)
    reservoir before and after refinement (seed 1) for each fold and arm,
    used by the plot scripts. Only the FX ones are needed for the figures.
*.png
    the figures.

Checking the numbers in the paper
---------------------------------
In MATLAB, in this folder:

    report_fixed_nih
    report_ablation_nih

The first prints 73.1% -> 78.6% balanced accuracy, 10 of 10 patients improved,
Wilcoxon p = 0.002, McNemar 312 repaired / 169 broken, p = 6.9e-11. The second
prints the ablation table.

Rerunning the training
----------------------
    run_hero_nih10_ablation
    report_fixed_nih
    report_ablation_nih

Each network is started with rng(1000*fold + seed), so the run is deterministic
and gives the same result files again. To run only the baseline and HERO-SNN,
set cfg.arms = {'noHERO','HERO'} at the top of the script. Training uses 40
epochs per class from each of the 9 training patients (720 per fold); testing
uses all epochs of the held-out patient. It takes about XX min per fold and
arm on <machine>. The script saves checkpoints (ckpt_*.mat) so it can be
restarted if interrupted; you do not need them to start.

Written and run in MATLAB <2025b>. Octave also runs it but its random
numbers differ, so the per-fold values will not be identical.

Data source: Li, A. et al. (2023). Epilepsy-iEEG-Multicenter-Dataset, OpenNeuro
ds003029, v1.0.7, doi:10.18112/openneuro.ds003029.v1.0.7.

Contact: <mgholami@aut.ac.nz>
