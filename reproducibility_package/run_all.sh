#!/bin/sh
# Full pipeline (about 1 h on 2 cores). Results land in ../results, tables/macros in ../paper.
set -e; cd "$(dirname "$0")/code"
python3 audit.py
for p in grouped dedup naive; do python3 run_cv.py $p 10; done
python3 orig_holdout.py; python3 train_gap.py
python3 nested_cv.py dedup 5; python3 nested_cv.py grouped 5; python3 nested_stats.py
python3 feat_ablation.py; python3 explain.py; python3 stats_calib.py
python3 ablation_stack.py 0 10; python3 seed_sweep.py; python3 ale_corr.py; python3 holdout_figures.py
python3 run_extra.py grouped 10; python3 run_extra.py dedup 10      # FT-Transformer rows (needs torch)
python3 stats_calib.py; python3 build_tables.py; python3 make_figures.py
python3 perclass_diffs.py; python3 conflict_errors.py                                          # error analysis by label consistency (Section "Where the errors fall")
python3 ../figure_scripts/fig_overview.py; python3 ../figure_scripts/figs_results.py; python3 ../figure_scripts/fig_lime_local.py   # Fig. 1 and the result figures -> ../../figs
# TabPFN: run tabpfn/tabpfn_kaggle.py on Kaggle, copy its three output files into results/, then: python3 integrate_tabpfn.py
