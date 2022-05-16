import argparse, os, sys, re
from os.path import splitext, basename
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import ast
import numpy as np
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
import helpers as nhp
from helpers import get_FRET_efficiency

color_dict = {
    'C': '#fcb040',
    'K': '#7f3f98'
}

def str2list(line, id_str):
    return np.array(ast.literal_eval(line.replace(id_str, '')))

def plot_histogram(summary_dict, fn):
    out_list = []
    bins = np.arange(0, 1.01, 0.01)
    nb_tagged_resn = len(summary_dict)
    fig, ax = plt.subplots(nb_tagged_resn, 1, figsize=[8.25, 2.91 * nb_tagged_resn])
    if not type(ax) == np.ndarray: ax = [ax]
    for ri, resn in enumerate(summary_dict):
        out_list.append(summary_dict[resn])
        ax[ri].hist(summary_dict[resn], bins=bins, color=color_dict[resn])
        ax[ri].set_xticks([0.25,0.5,0.75,1.0])
        ax[ri].set_xlim([0, 1])
    ax[-1].set_xlabel('$FRET (E)$')
    plt.tight_layout()
    fig.savefig(fn)
    plt.close(fig)
    plt.clf()
    return np.concatenate(out_list)


parser = argparse.ArgumentParser(description='Takes pdb files and produces histogram of FRET values')
parser.add_argument('--wd', type=str, required=True,
                    help='working directory')
parser.add_argument('--tagged-resn', required=True, type=str, nargs='+')
parser.add_argument('--detection-limit', type=float, default=0.20,
                    help='Minimum fret efficiency still considered readable.')
parser.add_argument('--pdb-dir', type=str, required=True, nargs='+',
                    help='directory containing lattice model pdb files.')
parser.add_argument('--pattern', type=str, default='^.+(?=_[0-9]+)',
                   help='Treat all names that adhere to the same regex pattern as different models of the same'
                        'peptide.')
parser.add_argument('--dist-based', action='store_true')
parser.add_argument('--event-duration', type=int, default=-1,
                    help='Define how many snapshots go in 1 event. provide -1 to get 1 event per structure '
                         '[default: -1]')
parser.add_argument('--max-events', type=int, default=np.inf,
                    help='Maximum number of events per molecule to take. [default: infinite]')


args = parser.parse_args()

pdb_list = nhp.parse_input_dir(args.pdb_dir, '*.pdb')
wd = nhp.parse_output_dir(args.wd, clean=True)
svg_dir = nhp.parse_output_dir(wd+'svg')
txt_dir = nhp.parse_output_dir(wd+'txt')

all_dict = {}
fp_dict = {}
fp_dist_dict = {}
e_dict = {}

for pdb_fn in pdb_list:
    pdb_id = splitext(basename(pdb_fn))[0]
    # try:
    with open(pdb_fn, 'r') as fh:
        efret_fp, dist_fp, energy_fp = None, None, None
        while efret_fp is None or dist_fp is None or energy_fp is None:
            line = fh.readline()
            if '1 FINGERPRINT' in line:
                efret_fp = str2list(line, 'REMARK   1 FINGERPRINT ')
            elif '1 DIST_FINGERPRINT' in line:
                dist_fp = str2list(line, 'REMARK   1 DIST_FINGERPRINT ')
            elif '1 ENERGIES' in line:
                energy_fp = str2list(line, 'REMARK   1 ENERGIES ')
    if args.event_duration == -1:
        nb_events, event_duration = 1, len(energy_fp)
    else:
        nb_events, event_duration = min(args.max_events, len(energy_fp) // args.event_duration), args.event_duration

    summary_dict = {resn: [] for resn in args.tagged_resn}
    batch_idx = np.array_split(np.arange(len(energy_fp)), nb_events)
    if len(batch_idx[-1]) < event_duration: batch_idx = batch_idx[:-1]
    for bi in batch_idx:
        cfpo_efret, cfpo_dist = efret_fp[bi], dist_fp[bi]
        for resn in args.tagged_resn:
            if args.dist_based:
                cur_dist_values = [x.get(resn, []) for x in cfpo_dist]
                if not len(cur_dist_values): continue
                summary_dict[resn].extend(get_FRET_efficiency(np.array(cur_dist_values).mean(axis=0)))
            else:
                cur_efret_values = [x.get(resn, []) for x in cfpo_efret]
                if not len(cur_efret_values): continue
                if not len(np.unique([len(ce) for ce in cur_efret_values])) == 1: continue
                summary_dict[resn].extend(np.array(cur_efret_values).mean(axis=0))
    all_dict[pdb_id] = summary_dict
    hist_vec = plot_histogram(summary_dict, f'{svg_dir}{pdb_id}.svg')
    np.savetxt(f'{txt_dir}{pdb_id}.txt', hist_vec)

pdb_id_dict = {}
for pdb_id in all_dict:
    x = re.search(args.pattern, pdb_id).group(0)
    pdb_id_dict[x] = pdb_id_dict.get(x, []) + [pdb_id]

fused_dict = {meta_id: {resn: [] for resn in args.tagged_resn} for meta_id in pdb_id_dict}
for meta_id in pdb_id_dict:
    for pdb_id in pdb_id_dict[meta_id]:
        cd = all_dict[pdb_id]
        for resn in cd:
            fused_dict[meta_id][resn].extend(cd[resn])

for meta_id in fused_dict:
    hist_vec = plot_histogram(fused_dict[meta_id], f'{svg_dir}{meta_id}.svg')
    np.savetxt(f'{txt_dir}{meta_id}.txt', hist_vec)
