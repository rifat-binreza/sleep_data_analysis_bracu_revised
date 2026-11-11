# Running TabPFN on Kaggle and folding it into the paper

`tabpfn_kaggle.py` evaluates TabPFN (version 2 weights, default settings, no tuning) on exactly the folds
used for every other model: the fold of every record is embedded in the script.

## On Kaggle
1. kaggle.com -> Create -> New Notebook.
2. Right panel -> Add Input -> search "Sleep Health and Lifestyle Dataset" (owner uom190346a) -> add it.
3. Right panel -> Session options -> Internet: On (needs a phone-verified account). Accelerator: GPU T4 if
   available; CPU also works, it is only slower.
4. Delete the default cell, paste the whole of `tabpfn_kaggle.py` into one cell, press Run.
5. It prints "repeat k/10 done" twenty times, then FINISHED and a short summary.
6. Right panel -> Output -> download `tabpfn_oof_grouped.csv`, `tabpfn_oof_dedup.csv`, `tabpfn_meta.json`.

If the script stops with "This CSV differs from the file used in the manuscript", upload `data/sleep.csv`
(your own copy) as a private Kaggle dataset, add it as Input instead, and run again.
If the weight download asks for a licence or token, the package has changed its defaults; send the error text.

## On your computer
1. Copy the three downloaded files into `reproducibility_package/results/`.
2. Put the dataset at `reproducibility_package/data/sleep.csv`.
3. `python reproducibility_package/code/integrate_tabpfn.py`
   This writes `paper_tables/tabpfn_numbers.tex` and redraws the figures that list all models.
4. The TabPFN rows and sentences are written to `paper_tables/tabpfn_numbers.tex`.
