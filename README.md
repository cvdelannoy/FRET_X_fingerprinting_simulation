# FRET X fingerprinting simulation
Simulate single-molecule protein fingerprinting using FRET X (Förster Resonance Energy Transfer via 
 DNA eXchange), train a classifier on them and evaluate the classifier's performance. Code in this 
 repository can be used to reproduce results described in our paper: ["Evaluation of FRET X for Single-Molecule 
Protein Fingerprinting"](https://www.biorxiv.org/content/10.1101/2021.06.30.450512v1).

### How to run
Commands assume you start in the root directory of this repository.

Code has been run on Ubuntu 18.04/20.04, with miniconda3. First install the conda environment:
```shell
conda env create -f env.yaml
```

1. The simulation routine reads out lattice starting structures from customary `npz` files. To convert fully-atomistic 
structures in pdb-format to `npz` files, run:
```shell
python tools/pdb2lattice.py --in-dir dir/containing/pdb_files  \  # <-- change directory
                            --out-dir npz/output/directory \     # <-- change directory
                            --cm-type ca
```

2. To label the N-terminus and any given combination of residue types - for example, cysteine (C), lysine (K) and 
both (CK) - we need to attach DNA-tags and optimize the altered lattice model: 
```shell
python generate_lattice_models_sm.py --in-dir npz/output/directory  \     # <-- change directory
                                     --out-dir models/output/directory  \ # <-- change directory
                                     --tagged-resn C K CK  \
                                     --labeling-model perfect \
                                     --accomodate-tags  \
                                     --nb-processes 2 \                  # number of processes == number of temp chains
                                     --max-cores 24                       # <-- adapt to your machine
```
Models are parallel tempered, so multiple models of the same structure are optimized simultanously at different
model temperatures. More chains will allow a finer exploration of the energy landscape, but be aware that each chain
engages an extra core, so this eats resources quickly!

`--accomodate-tags` will ensure that lattice model optimization is continued until sufficient space is available for
all DNA-tags.

`--labeling-model` controls whether labeling errors are introduced or not. Choose `perfect` for no errors, or `default`
for error rates as used in our paper.

3. Collect pdb-files of completed models:
```shell
python tools/collect_completed_runs.py \
            --in-dirs models/output/directory/tagC \     # <-- change directory
                      models/output/directory/tagK \
                      models/output/directory/tagCK \
            --out-dir collected_models/output/directory \  # <-- change directory
            --success-txt success_models.txt \             
            --max-fold 10
```
As it is possible that models sometimes fail to optimize far enough to include all DNA-tags, this step ensures 
that only proteins for which a certain minimum number of models (controlled by `--max-fold`) is kept.


4. Fingerprints can now be predicted for the generated models, after which an SVM classifier can be trained and tested 
on them in a cross-validation scheme. To use the fingerprint classification routine, manually rename the directories
of collected models to include the labeling model, as follows: `tagXXX_labmodX`, 
for example: `tagCK_labmodPerfect`. Then run the classification and evaluation pipeline as follows:
```shell
python fingerprint_classification/run_fp_classification_pipeline.py \
                --in-dirs tagX_labmodX tagY_labmodX tagZ_labmodX \   # <-- change directories
                --cores 24 \                                         # <-- adapt to your machine
                --original-dir npz/output/directory \                # <-- original npz file directory from step 1
                --out-dir results/output/directory                   # <-- change directory 
``` 
Figures illustrating the performance of the SVM classifier on your fingerprints can now be found in the specified 
directory.

After classification fingerprints may also be graphed:
```shell
python fingerprint_classification/graph_fingerprints.py
                --

```

### Recreating paper figures
To recreate fingerprints of 40-residue model peptides, BCL-like 2 isoforms and PTGS1 isoforms, follow the above 
procedure using the respective `npz` files in `data/`. E.g. to reproduce the PTGS1 fingerprints, from the repo's root 
directory, first construct lattice models and collect the produced PDB files:

```shell
python generate_lattice_models_sm.py --in-dir data/PTGS1_isoforms_npz/  \    
                                     --out-dir data/PTGS1_isoforms_lat/  \ 
                                     --tagged-resn C CK \
                                     --labeling-model perfect \
                                     --nb-processes 2 \
                                     --max-cores 24                       # <-- adapt to your machine

python tools/collect_completed_runs.py \
            --in-dirs data/PTGS1_isoforms_lat/tagC \
                      data/PTGS1_isoforms_lat/tagCK \
            --out-dir data/PTGS1_isoforms_fps \
            --success-txt data/PTGS1_isoforms_success_models.txt \             
            --max-fold 10

```

Then parse the fingerprints from the PDB files and graph them:
```
python fingerprint_classification/parse_fingerprints.py \
            --in-dir \
            --original-dir \
            --tagged-resn C K \
            --efret-resolution 1 \
            --out-pkl data/PTGS1_isoforms_fps_res1.pkl

python fingerprint_classification/graph_mass_fingerprints.py \
            --fp-pkl data/PTGS1_isoforms_fps_res1.pkl \
            --out-svg data/PTGGS1_fingerprints.svg
```

To recreate the figures on >300 UniProt proteins in our paper, first download the additional data (~6GB):
```
./download_uniprot_data.sh 
```
The new folder `data/FRET_X_fingerprinting_data/` contains `npz` files, ready to be optimized, as in step 2, and 
finished lattice model pdb-files, which may serve as input for step 4.
