# FRET X proteoform fingerprinting
Code and data in this directory can be used to reproduce results described in our paper on FRET X fingerprinting of proteoforms: ["Full-Length Single-Molecule Protein Fingerprinting"](https://www.biorxiv.org/content/10.1101/2023.09.26.559471v1).

### Prepare environment
Code has been tested on Ubuntu 20.04, with miniconda3 and `mamba` drop-in. 

Construct and initiate the required conda environment as follows (with the `mamba` drop-in replacement, otherwise use `conda`):
```shell
mamba env create -f FRET_X_proteoform_conda_env.yaml
conda activate FRET_X_proteoform
```

### Data parsing
First, BCD text files as returned by the [rene](https://github.com/kahutia/transient_FRET_analyzer2) tool needs to be converted to a simple 1-molecule-per-line `txt` format. Only retains molecules that exhibit a single or no donor-only FRET peak (i.e. where only a donor fluorescent molecule is present for calibration, see optional arguments): 
```shell
python asyn_classifier/parse_txt_files.py \
    --in-dir path/to/rene/data \
    --out-dir path/to/save/txt/files
```
Optional arguments:
- `--fp-len`: Expected number of values in fingerprint. If number deviates for a molecule, omit it. [default: 1]
- `--cutoffs`: Upper and lower detection limit for FRET efficiencies. [default: 0.1 0.9]
  - If FRET > upper limit: remove value.
  - If FRET < lower limit: label it as donor only peak. Will be used to check for donor-only peak presence and then removed.
- `--out-fmt`: output format, may be `txt` or `pkl` (deprecated). [default: `txt`]

This returns the following:
- A single output file in the output folder per input file, bearing the same name as the input file. 

To generate a simulated dataset of molecules labeled at both termini, we can combine txt files for N- and C-terminally labeled molecules at random as follows:
```shell
python asyn_classifier/combine_NC_term_fret.py \
    --n-term path/to/nterm/txt/files \
    --c-term path/to/cterm/txt/files \ 
    --out-dir path/to/save/txt/files
```


### Classifiers for aSyn recognition
From this directory, an SVM for smFRET values from aSyn mutants can be trained and tested as follows:
```shell
python asyn_classifier/asyn_classifier_train.py \
    --train-dir path/to/training/data \
    --test-dir path/to/test/data \
    --pred-csv path/to/output_csv.csv
```
Optional arguments:
- `--regex`: Regular expression which will be applied to input files, to discern true label for each input file if possible. By default set correctly to parse aSyn mutant identity for files provided in this repo.

This returns the following:
- [csv_name].csv: details per-molecule FRET value(s), predicted class and true class (if available). 
- [csv_name]_counts.svg: relative fractions of each mutant, with bootstrapped confidence intervals.
- [csv_name]_bootstrapped_fractions.csv: bootstrapped predicted fractions, used to generate [csv_name]_counts.svg
- [csv_name]_hist.svg: FRET histogram colored by assigned mutant.

### k-fold cross-validation of aSyn classifier

Automatically repeat the fitting and testing procedure k times in a k-fold cross validation procedure:
```shell
python asyn_classifier/asyn_classifier_pipeline.py \
    --train path/to/train/data \
    --test path/to/test/data \
    --out-dir path/to/results
```
Optional arguments:
- `--regex`: Regular expression which will be applied to input files, to discern true label for each input file if possible. By default set correctly to parse aSyn mutant identity for files provided in this repo.
- `--nb-folds`: Number of folds to split up data [default: 10]
- `--nb-cores`: Cores to engage simultaneously, more = faster running [default: 4]
- `--sort-asyn`: Sort aSyn mutants in output figures on position [default: no sorting]
- `--order`: manually define order of classes in output figures, overrules `--sort-asyn` [default: no sorting]

This returns the following:
- `folds`: A directory containing outputs as given by `asyn_classifier_train.py` for each fold
- `heatmap_confmat_[color].svg`: The confusion matrix for all classes in several colors
- `precision_recall.svg`: A precision-recall plot detailing performance for each class
-  `stats.yaml`: Overall accuracy over all classes, all folds

### Reproducing figures
aSyn classification results and figures can be reproduced using the data in this directory. The following is provided in the `data` subdirectories:
- `rene`: molecule analysis files in rene BCD format
- `txt`: rene files parsed into 1-molecule-per-line text format
- `runs`: analysis runs with default arguments:
  - 20230812_classifier_pipeline_Cterm: k-fold cross validation run, mutants measured from C-terminal.
  - 20230812_classifier_pipeline_Nterm: k-fold cross validation run, mutants measured from N-terminal.
  - 20230816_classifier_pipeline_CNterm: K-fold cross validation run, simulated dataset of molecules mesaured from both termini.
  - 20230816_mixture_classification: single classifier (`asyn_classifier_train.py`) runs for equimolar mixture experiments of 4 mutants.
