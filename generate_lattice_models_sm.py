import pathlib
import re
import argparse
import yaml
from Bio.PDB import PDBParser
from Bio import SeqIO
import os
import numpy as np
import pandas as pd
# import faulthandler
import helpers as nhp
from LatticeModelComparison import LatticeModelComparison
from ParallelTempering import ParallelTempering
from jinja2 import Template
from os.path import basename, splitext
import snakemake as sm
from shutil import copyfile, rmtree
from itertools import product

pdb_parser = PDBParser()
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

parser = argparse.ArgumentParser(description='Generate cubic lattice models for a (list of) pdb names or AA seqs')
parser.add_argument('--in-dir', type=str, nargs='+',
                    help='entity/fasta file or directory containing entity/fasta files of which to generate models')
# --- model parameters ---
parser.add_argument('--temp-range', type=float, nargs=2, default=[0.01, 0.001],
                    help='Use a range of equally spaced temperatures between given values in a parallel tempering search.')
parser.add_argument('--tagged-resn', type=str, default=['None'], nargs='+',
                    help='Define residue(s) with 1-letter-code that should be replaced with a very hydrophillic tag.'
                         'May define multiple combinations. Set to None for no tagged resn [default: None]')
parser.add_argument('--experimental-mode', type=int, default=0,
                    help='Several experimental settings under 1 switch, numeric.')
parser.add_argument('--cm-pdb-dir', type=str, default=[], nargs='+',
                    help='center-of-mass pdbs directory, for finetuning of starting structures. Necessary if --finetune-structure is on')
parser.add_argument('--finetune-structure', action='store_true',
                    help='Minimize lattice model RMSD w.r.t. center-of-mass structure provided through cm-pdb-dir by the same ID')
parser.add_argument('--labeling-model', type=str, default='perfect',
                    help='Yaml file containing labeling probability for each residue, for each labeling chemistry '
                         '(no value == no labeling). Supply "standard" for regular model or "perfect" for '
                         'no mislabeling or path to yaml [default: perfect].')
# --- lattice properties ---
parser.add_argument('--lattice-type', type=str, default='bcc', choices=['cubic', 'bcc'],
                    help='Set type of lattice to use. Choices: cubic, bcc [ default: bcc]')
# --- model iterations parameters ---
parser.add_argument('--nb-steps', type=int, default=1,
                    help='number of mutations to perform at each MC iteration.')
parser.add_argument('--iters-per-tempswap', default=100, type=int,
                    help='If using parallel tempering, define per how many rounds a temperature swap should be performed')
parser.add_argument('--mc-iters', type=int, default=500,
                    help='number Monte Carlo iterations to perform for each model.')
parser.add_argument('--nb-models', type=int, default=10,
                    help='number of models to create per AA sequence.')
parser.add_argument('--nb-processes', type=int, default=4,
                    help='Define how many processes to engage at once in parallel tempering.')
parser.add_argument('--no-regularization', action='store_true',
                    help='Do not add regularization term to energy function.')
parser.add_argument('--accomodate-tags', action='store_true',
                    help='Run MC iterations until tag penalty is 0.')
parser.add_argument('--max-accomodate-rounds', type=int, default=5,
                    help='If accomodating tags, number of rounds to run before re-initiating.')
parser.add_argument('--free-sampling', action='store_true',
                    help='Do not optimize structure during snapshot generation, accept all')
# --- Result saving options ---
parser.add_argument('--out-dir', type=str, required=True,
                    help='Location where model pdb files are stored.')
parser.add_argument('--snapshots', nargs=2, type=int, default=[0,0],
                    help='If given [n,s], saves n snapshots with s steps in between after convergence/end of run.')
parser.add_argument('--save-intermediate-structures', action='store_true',
                    help='Save structure after temperature swaps in this pdb file')
parser.add_argument('--store-energies', action='store_true',
                    help='Store base energy and individual contributions to energy in tsv file')
parser.add_argument('--max-cores', type=int, default=4)
parser.add_argument('--dry-run', action='store_true')

args = parser.parse_args()

out_dir = nhp.parse_output_dir(args.out_dir)
log_dir = nhp.parse_output_dir(out_dir +'logs')
ent_list = nhp.parse_input_dir(args.in_dir, '*.npz')
pdb_id_list = [splitext(basename(ent))[0] for ent in ent_list]
pdb_id_list_str = ','.join(pdb_id_list)
cm_pdb_list = ','.join(args.cm_pdb_dir)

# Copy input files to destination folder
in_dir = nhp.parse_output_dir(out_dir + 'in_npz')
for ent in ent_list:
    copyfile(ent, in_dir+basename(ent))

def get_mod_tuples(pdb_id_list, nb_models, tagged_resn, out_dir):
    out_list = []
    for tup in product(pdb_id_list, list(range(nb_models)), tagged_resn):
        if not os.path.exists(f'{out_dir}tag{tup[2]}/{tup[0]}/{tup[0]}_{tup[1]}.pdb'): out_list.append(tup)
    return out_list

mod_tuples = get_mod_tuples(pdb_id_list, args.nb_models, args.tagged_resn, out_dir)

if args.accomodate_tags:
    with open(f'{__location__}/generate_lm_accomodate_tags.sf', 'r') as fh: template_txt = fh.read()
    accomodate_tags_rounds = 0
    unfinished_ids_fn = f'{out_dir}unfinished_ids.txt'
    # mod_tuples = list(product(pdb_id_list, list(range(args.nb_models)), args.tagged_resn))
    while len(mod_tuples):
        with open(unfinished_ids_fn, 'w') as fh:
            fh.write('')
        sf_txt = Template(template_txt).render(
            __location__=__location__,
            mod_tuples=mod_tuples,
            in_dir=in_dir,
            out_dir=out_dir,
            max_nb_models=args.nb_models,
            processes=args.nb_processes,
            temp_min=args.temp_range[0], temp_max=args.temp_range[1],
            iters_per_tempswap=args.iters_per_tempswap,
            mc_iters=args.mc_iters,
            nb_steps=args.nb_steps,
            lattice_type=args.lattice_type,
            nb_snapshots=args.snapshots[0], snapshot_dist=args.snapshots[1],
            store_energies=args.store_energies,
            save_intermediate_structures=args.save_intermediate_structures,
            experimental_mode=args.experimental_mode,
            cm_pdb_str=str(cm_pdb_list).strip('[]').replace(',', ''),
            finetune_structure=args.finetune_structure,
            no_regularization=args.no_regularization,
            labeling_model=args.labeling_model,
            accomodate_tags=args.accomodate_tags,
            free_sampling=args.free_sampling
        )
        sf_fn = f'{out_dir}generate_lm{accomodate_tags_rounds}.sf'
        with open(sf_fn, 'w') as fh: fh.write(sf_txt)
        sm.snakemake(sf_fn, cores=args.max_cores, keepgoing=True, dryrun=args.dry_run)

        # --- reload model tuples that did not end with all tags accomodated ---
        with open(unfinished_ids_fn, 'r') as fh:
            mod_tuples = [tup.strip().split('\t') for tup in fh.readlines()]
        accomodate_tags_rounds += 1
        if not accomodate_tags_rounds % args.max_accomodate_rounds:
            for tup in mod_tuples:
                os.remove(f'{out_dir}tag{tup[2]}/{tup[0]}/{tup[0]}_{tup[1]}_unoptimizedTags.npz')
else:
    with open(f'{__location__}/generate_lm_accomodate_tags.sf', 'r') as fh: template_txt = fh.read()
    sf_txt = Template(template_txt).render(
        __location__=__location__,
        mod_tuples=mod_tuples,
        in_dir=in_dir,
        out_dir=out_dir,
        max_nb_models=args.nb_models,
        processes=args.nb_processes,
        temp_min=args.temp_range[0], temp_max=args.temp_range[1],
        iters_per_tempswap=args.iters_per_tempswap,
        mc_iters=args.mc_iters,
        nb_steps=args.nb_steps,
        lattice_type=args.lattice_type,
        nb_snapshots=args.snapshots[0], snapshot_dist=args.snapshots[1],
        store_energies=args.store_energies,
        save_intermediate_structures=args.save_intermediate_structures,
        experimental_mode=args.experimental_mode,
        cm_pdb_str=str(cm_pdb_list).strip('[]').replace(',', ''),
        finetune_structure=args.finetune_structure,
        no_regularization=args.no_regularization,
        labeling_model=args.labeling_model,
        accomodate_tags=args.accomodate_tags,
        free_sampling=args.free_sampling
    )
    sf_fn = f'{out_dir}generate_lm.sf'
    with open(sf_fn, 'w') as fh:
        fh.write(sf_txt)
    sm.snakemake(sf_fn, cores=args.max_cores, keepgoing=True, dryrun=args.dry_run)
