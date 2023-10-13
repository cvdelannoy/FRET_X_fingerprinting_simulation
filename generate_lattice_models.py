import re
import argparse
import yaml
from Bio.PDB import PDBParser
from Bio import SeqIO
import os
import numpy as np
import pandas as pd
import helpers as nhp
from pathlib import Path

from LatticeModelComparison import LatticeModelComparison
from ParallelTempering import ParallelTempering

pdb_parser = PDBParser()
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

parser = argparse.ArgumentParser(description='Generate cubic lattice models for a (list of) pdb names or AA seqs')
# --- IO ---
parser.add_argument('--in-dir', type=str,
                    help='entity/fasta file or directory containing entity/fasta files of which to generate models')
parser.add_argument('--cm-pdb-dir', type=str, default=None,
                    help='center-of-mass pdbs directory, for finetuning of starting structures. '
                         'Necessary if --finetune-structure is on')
parser.add_argument('--finetune-structure', action='store_true',
                    help='Minimize lattice model RMSD w.r.t. center-of-mass structure provided through cm-pdb-dir by '
                         'the same ID')
# --- model parameters ---
parser.add_argument('--temp-range', type=float, nargs=2, default=[0.01, 0.001],
                    help='Use a range of equally spaced temperatures between given values in a parallel tempering search.')
parser.add_argument('--tagged-resn', type=str, default='',
                    help='Define residue(s) with 1-letter-code that should be replaced with a very hydrophillic tag.')
parser.add_argument('--pairs-mat', type=str, default=f'{__location__}/potential_matrices/aa_water2_abeln2011.txt',
                    help='Matrix with pair potentials for interactions between residues and residues vs solvents.')
parser.add_argument('--labeling-model', type=str, default='perfect',
                    help='Yaml file containing labeling probability for each residue, for each labeling chemistry '
                         '(no value == no labeling). Supply "standard" for regular model or "perfect" for '
                         'no mislabeling or path to yaml [default: perfect].')
parser.add_argument('--fretxy', action='store_true',
                    help='Run a FRET XY simulation instead.')
parser.add_argument('--experimental-mode', type=int, default=0,
                    help='Several experimental settings under 1 switch, numeric.')
# --- lattice properties ---
parser.add_argument('--lattice-type', type=str, default='bcc', choices=['cubic', 'bcc', 'bcc_old'],
                    help='Set type of lattice to use. Choices: cubic, bcc [ default: bcc]')
parser.add_argument('--starting-structure', type=str,
                    choices=['free_random', 'anchored_random', 'stretched', 'serrated'], default='stretched',
                    help='If no starting structure is provided, choose how to initialize structure [default: stretched]')


# --- model iterations parameters ---
parser.add_argument('--nb-steps', type=int, default=1,
                    help='number of mutations to perform at each MC iteration.')
parser.add_argument('--accomodate-tags', action='store_true',
                    help='Run MC iterations until tag penalty is 0.')
parser.add_argument('--iters-per-tempswap', default=1000, type=int,
                    help='If using parallel tempering, define per how many rounds a temperature swap should be performed')
parser.add_argument('--mc-iters', type=int, default=1000,
                    help='number Monte Carlo iterations to perform for each model.')
parser.add_argument('--nb-models', type=int, default=5,
                    help='number of models to create per AA sequence.')
parser.add_argument('--nb-processes', type=int, default=4,
                    help='Define how many processes to engage at once in parallel tempering.')
parser.add_argument('--early-stopping-rounds', type=int, default=-1,
                    help='Number of consecutive temperature swap rounds without improvement in energy'
                         'before training is stopped. Supply -1 to disable early stopping.')
parser.add_argument('--restart', action='store_true',
                    help='Restart previous run; if folder with structures carrying same id exists, start with'
                         'next model index. If not, start with 0.')
parser.add_argument('--start-idx', type=int,
                    help='Models should be generated starting with this index. Overrides --restart!')
parser.add_argument('--no-regularization', action='store_true',
                    help='Do not add regularization term to energy function.')
parser.add_argument('--free-sampling', action='store_true',
                    help='Do not optimize structure during snapshot generation, accept all')

# --- Result saving options ---
parser.add_argument('--out-dir', type=str, required=True,
                    help='Location where model pdb files are stored.')
parser.add_argument('--store-rg', type=str, choices=['off', 'tswap', 'full'], default='tswap', required=False,
                    help='Store the estimated radius of gyration at each training step in the produced pdb.')
parser.add_argument('--save-all-pdbs', action='store_true',  # todo superfluous, remove
                    help='Save all pdbs at intermediate steps as well, instead of keeping only the eventual best pdb')
parser.add_argument('--snapshots', nargs=2, type=int, default=[0,0],
                    help='If given [n,s], saves n snapshots with s steps in between after convergence/end of run.')
parser.add_argument('--save-intermediate-structures', action='store_true',
                    help='Save structure after temperature swaps in this pdb file')
parser.add_argument('--store-energies', action='store_true',
                    help='Store base energy and individual contributions to energy in tsv file')

# --- Temporary debugging options ---
parser.add_argument('--equidist-temps', action='store_true',
                    help='Temperatures equidistance spaced in T range (i.o. beta range).')


args = parser.parse_args()

if args.out_dir[-1] != '/': args.out_dir += '/'

pairs_mat = nhp.get_pairs_mat(args.pairs_mat)
ent_list = nhp.parse_input_dir(args.in_dir, '*.npz')
# ent_list = nhp.parse_input_dir(args.in_dir, '*.ent')
# if len(ent_list) == 0: ent_list = nhp.parse_input_dir(args.in_dir, '*.fasta')
# if len(ent_list) == 0: ent_list = nhp.parse_input_dir(args.in_dir, '*.pdb')
# if len(ent_list) == 0: ent_list = nhp.parse_input_dir(args.in_dir, '*.npz')
if len(ent_list) == 0: raise ValueError('No npz files found')

# Load a labeling model, describing the probability of labeling a given residue for a given labeling chemistry
# Default: perfect labeling
labeling_model = args.labeling_model
if labeling_model == 'standard':
    labeling_model = f'{__location__}/data/labeling_models/standard.yml'
elif labeling_model == 'perfect':
    labeling_model = f'{__location__}/data/labeling_models/perfect.yml'
with open(labeling_model, 'r') as fh: labeling_model = yaml.load(fh, yaml.SafeLoader)

# Load cm pdbs, if available
cm_dict = {}
if args.cm_pdb_dir is not None:
    cm_list = nhp.parse_input_dir(args.cm_pdb_dir, '*.pdb')
    cm_dict = {os.path.splitext(os.path.basename(cmd))[0].split('_')[0]:
                   np.vstack([atm.coord for atm in pdb_parser.get_structure('cm', cmd).get_atoms() if atm.name == 'CA'])
               for cmd in cm_list}
elif args.finetune_structure:
    raise ValueError('Must provide --cm-pdb-dir if finetuning structure.')

# Loop over protein files
for ent in ent_list:
    pdb_id, ext = os.path.splitext(os.path.basename(ent))
    out_dir_pdb = f'{args.out_dir}{pdb_id}/'
    Path(out_dir_pdb).mkdir(parents=True, exist_ok=True)
    model_idx_start = 0
    nb_models = args.nb_models  # For each protein [nb_models] model runs are performed

    if args.finetune_structure:
        if pdb_id.split('_')[0] not in cm_dict:
            print(f'Cannot find cm structure {pdb_id} for polishing, skipping...')

    if args.start_idx:  # Start counting at a different index than 0 - only used for snakemake implementation
        model_idx_start = args.start_idx
        nb_models = model_idx_start + nb_models
    elif args.restart:  # Allow restarting an interrupted run, by checking final output pdb for a given index
        existing_fn_list = nhp.parse_input_dir(out_dir_pdb, pattern='*.pdb')
        highest_mod_idx = -1
        for pdb_fn in existing_fn_list:
            re_obj = re.search(f'(?<={pdb_id}_)[0-9]+(?=.pdb)', pdb_fn)
            if re_obj: highest_mod_idx = max(highest_mod_idx, int(re_obj.group(0)))
        if highest_mod_idx + 1 >= nb_models:
            print(f'Restart for {pdb_id} not required: model {model_idx_start} of {nb_models} was already produced.')
            continue
        model_idx_start = highest_mod_idx + 1

    tagged_resn = '' if args.tagged_resn == 'None' else args.tagged_resn
    with np.load(ent) as fh:
        aa_seq = fh['sequence']
        coords = fh['coords'].astype(int)
        ss_df = pd.DataFrame(fh['secondary_structure'], columns=['H', 'S', 'L'])
        if 'propka' in fh:
            propka_df = pd.DataFrame(fh['propka'], columns=['residue_id', 'reactivity', 'pKa', 'buried'])
        else:
            propka_df = pd.DataFrame(columns=['residue_id', 'reactivity', 'pKa', 'buried'])
        if 'acc_tagged_resi' in fh:
            acc_tagged_resi = fh['acc_tagged_resi'][()]
        else:
            acc_tagged_resi = 0

    if args.accomodate_tags and os.path.exists(f'{out_dir_pdb}{pdb_id}_{args.start_idx}_unoptimizedTags.npz'):
        is_retry = True
        print(f'Reloading previous attempt at tag accomodation')
        with np.load(f'{out_dir_pdb}{pdb_id}_{args.start_idx}_unoptimizedTags.npz', allow_pickle=True) as fh:
            # aa_seq_original = aa_seq
            aa_seq = fh['sequence']
            coords = fh['coords'].astype(int)
            ss_df = pd.DataFrame(fh['secondary_structure'], columns=['H', 'S', 'L'])
            tagged_resi = fh['tagged_resi'][()]
            # tagged_resi = {tr:[] for tr in fh['tagged_resn']}
            # for ri, resn in enumerate(aa_seq):
            #     if resn == 'TAG': tagged_resi[aa_seq_original[ri]].append(ri)
                # tagged_resi = {: [ri for ri, resn in enumerate(aa_seq) if resn == 'TAG']}
    else:
        is_retry = False

    # Set temperatures at which parallel-tempered models are run
    if args.equidist_temps:
        # option 1: temperatures are linearly spaced as temperatures and then converted to beta-values
        temp_list = np.linspace(min(args.temp_range), max(args.temp_range), num=args.nb_processes)
        beta_list = 0.01 / temp_list
    else:
        # option 2: temperatures are linearly spaced as beta-values
        beta_min = 0.01 / max(args.temp_range)
        beta_max = 0.01 / min(args.temp_range)
        if args.experimental_mode == 8:
            beta_list = np.concatenate((np.linspace(beta_min, beta_max, num=args.nb_processes-1)[::-1], [np.nan]))
        else:
            beta_list = np.linspace(beta_min, beta_max, num=args.nb_processes)[::-1]

    # Construct model
    for ns in range(model_idx_start, nb_models):
        print(f'generating new model at {out_dir_pdb}{pdb_id}_{ns}.pdb ')
        if os.path.exists(f'{out_dir_pdb}{pdb_id}_{ns}.pdb'):
            print(f'skipping {pdb_id}_{ns}, already exists')
            continue

        print(f'{nhp.print_timestamp()} Generating model {ns+1} for {pdb_id} ')

        # Determine which residues are labeled. Here labeling model is applied (i.e. incomplete/mis-labeling takes place)
        if not is_retry:
            tagged_resi = nhp.get_tagged_resi(labeling_model, tagged_resn, aa_seq, acc_tagged_resi, propka_df, args.fretxy)
        for tri in tagged_resi:
            if args.nb_processes > 1:  # Run parallel tempering with [args.nb_processes] chains
                pt = ParallelTempering(
                    # --- model ID args ---
                    pdb_id=pdb_id,  # unique id for model
                    reactive_idx=tri,
                    model_nb=ns,    # Idx of this model run
                    # --- model run params ---
                    aa_sequence=aa_seq,                                # Amino acid sequence in 1-letter array
                    tagged_resn=tagged_resn,
                    beta_list=beta_list,                               # beta values at which to run parallel tempering chains (i.e. 0.01/T)
                    nb_processes=args.nb_processes,                    # Number of processes/cores to engage at once (== number of parallel tempering chains)

                    lattice_type=args.lattice_type,                    # Type of lattice to adhere to (e.g. cubic or bcc)
                    coords=coords,                                     # lattice CA starting coordinates, [L x 3] int np array, or None if starting coordinates should be generated
                    starting_structure=args.starting_structure,        # If no starting coords are provided, indicate how starting coords should be generated (e.g. 'stretched' or 'random')

                    nb_steps=args.nb_steps,                            # Number of mutations to apply at each MC iteration
                    nb_iters=args.mc_iters,                            # Total number of MC iterations
                    iters_per_tempswap=args.iters_per_tempswap,        # Number of MC iterations to perform before swapping models between parallel tempering chains

                    early_stopping_rounds=args.early_stopping_rounds,  # number of rounds in which no better model is accepted before stopping early, -1 to disable

                    experimental_mode=args.experimental_mode,
                    cm_coords=cm_dict.get(pdb_id.split('_')[0], None),
                    finetune_structure=args.finetune_structure,
                    # -- E modifiers ---
                    pairs_mat=pairs_mat,        # pandas df containing energy modifier for each combination of residue types + water interactions
                    secondary_structure=ss_df,  # secondary structure energy modifiers, [L x 3] pandas df with E modifier for H(elix)/S(trand)/L(oop) resp. in columns
                    tagged_resi=tagged_resi[tri],    # tagged residue indices
                    # --- output args ---
                    save_dir=out_dir_pdb,                                           # directory to save results to
                    store_rg=args.store_rg,                                         # Store radius of gyration at each temp swap ('tswap'), each MC iteration ('full') or not at all ('off')
                    snapshots=args.snapshots,                                       # Tuple [N, T] denoting to make N snapshots of final structure spaced by T MC iterations
                    save_intermediate_structures=args.save_intermediate_structures, # if True, save structure for each temp swap in separate file
                    store_energies=args.store_energies,                              # if True, store energy for each temp swap in separate file
                    no_regularization=args.no_regularization,

                    accomodate_tags=args.accomodate_tags,
                    free_sampling=args.free_sampling
                )
                pt.do_mc_parallel()
            else:  # If one process available, run single chain at lowest temperature
                beta = 0.01 / args.temp_range[0]
                intermediate_fn = None
                if args.save_intermediate_structures:
                    intermediate_fn = f'{out_dir_pdb}{pdb_id}_{ns}_intermediates.pdb'
                lmc = LatticeModelComparison(
                    pdb_id=pdb_id,
                    # --- model ID args ---
                    mod_id=ns,  # Idx for this model run
                    # --- model run params ---`
                    aa_sequence=aa_seq,  # Amino acid sequence in 1-letter array
                    beta=beta,           # Beta value at which to run chain (i.e. 0.01/T)

                    lattice_type=args.lattice_type,             # Type of lattice to adhere to (e.g. cubic or bcc)
                    coords=coords,                              # Lattice CA starting coordinates, [L x 3] int np array, or None if starting coordinates should be generated
                    starting_structure=args.starting_structure, # If no starting coords are provided, indicate how starting coords should be generated (e.g. 'stretched' or 'random')

                    nb_steps=args.nb_steps,  # Number of mutations to apply at each MC iteration
                    experimental_mode=args.experimental_mode,
                    # --- E modifiers ---
                    pairs_mat=pairs_mat,        # pandas df containing energy modifier for each combination of residue types + water interactions
                    secondary_structure=ss_df,  # secondary structure energy modifiers, [L x 3] pandas df with E modifier for H(elix)/S(trand)/L(oop) resp. in columns
                    tagged_resi=tagged_resi,    # tagged residue indices
                    # --- output args ---
                    store_rg=args.store_rg,  # Store radius of gyration at each temp swap ('tswap'), each MC iteration ('full') or not at all ('off')
                    stepwise_intermediates=intermediate_fn,
                    no_regularization=args.no_regularization
                )
                lmc.do_mc(args.mc_iters)
                lmc.make_snapshots(args.snapshots[0], args.snapshots[1], [lmc.best_model.rg],
                                   f'{out_dir_pdb}{pdb_id}_{ns}.pdb', free_sampling=args.free_sampling)
        Path(f'{out_dir_pdb}{pdb_id}_{ns}_done').touch()  # make file to signal snakemake run is completed (clunky!)
