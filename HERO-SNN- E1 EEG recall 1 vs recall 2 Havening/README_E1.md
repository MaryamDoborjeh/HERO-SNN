E1 - Havening EEG, Recall 1 vs Recall 2
=======================================

This folder has the code and results for experiment E1 of the HERO-SNN paper
(Neuroinformatics). The task is to classify two recall states of the same
distressing memory within each participant: Recall 1 (first recall at the start
of the session) and Recall 2 (second recall, after the distraction tasks of the
first cycle). 13 participants, 64-channel EEG, 2-s epochs, leave-one-subject-out.
Baseline is STDP + deSNN, HERO-SNN is the same pipeline plus the refinement stage.

This folder includes:
------------
run_hero_havening_fixed.m
    main script. Runs the baseline and the HERO-SNN arm for all 13 folds,
    5 seeds per fold, majority vote. The per-fold drift / eta / passes
    (Table 3b in the paper) are in the FIXED table at the top of the file.
report_fixed_havening.m
    reads the two result files and prints the per-participant table, pooled
    confusion matrices, Wilcoxon and McNemar (Table 1 and Table A2).
plot_snn_connections_FX.m, plot_hero_weights_pool_FX.m, plot_hero_consensus_FX.m
    make Fig. 4 and Fig. 5 from the snapshot files.
brain_coordinates_300.csv, eeg_mapping_biosemi64.csv
    the 300-neuron Talairach template and the 64 electrode positions.
prepost_final_results_all_full_FX_noHERO.mat / prepost_final_results_all_full_FX.mat
    results of the baseline and HERO-SNN runs (per fold and per seed predictions
    plus the parameters used). The numbers in the paper come from these two files.
prepost_final_all_full_FX_noHERO_log / prepost_final_all_full_FX_log
    the console output of those runs.
hero_snapshot_all_full_FX_f2 ... f13.mat
    reservoir before and after refinement (seed 1) for the 10 evaluated folds,
    used by the plot scripts.
*.png
    the figures.

The EEG data file (hero_dataset_havening_64ch_2s.mat) is not in this folder.
It is participant data from Nottingham Trent University and is available on
request, subject to NTU approval. Its format is X (512 x 64 x 182, time x
channels x epochs), y (1 = Recall 1, 2 = Recall 2) and pid (participant 1-13).

Checking the numbers in the paper
---------------------------------
You do not need the data for this. In MATLAB, in this folder:

    report_fixed_havening

This prints 70.7% -> 81.4% balanced accuracy, 8 of 10 participants improved and
2 unchanged, Wilcoxon p = 0.0078, McNemar 19 repaired / 4 broken, p = 0.0026.

Rerunning the training
----------------------
Put hero_dataset_havening_64ch_2s.mat in the folder and run

    run_hero_havening_fixed
    report_fixed_havening

Seeds are rng(1000*fold + seed), so the run is deterministic and gives the same
result files again. Folds 1, 9 and 11 are used for training only (signal
quality screening done beforehand); the other 10 are the evaluated participants.
It takes about XX min per fold and arm on <machine>. The script saves
checkpoints (ckpt_*.mat) so it can be restarted if interrupted; you do not need
them to start.

Written and run in MATLAB <R2025b>. 

Contact: Maryam Doborjeh <mgholami@aut.ac.nz>
