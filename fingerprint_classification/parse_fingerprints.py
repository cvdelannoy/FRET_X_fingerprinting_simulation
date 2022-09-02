import re, os, sys, argparse, warnings
from os.path import basename, dirname, splitext
from Bio.PDB import PDBParser
import numpy as np
import pickle
import ast
from collections import ChainMap

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])
from helpers import parse_input_dir, parse_output_dir, get_FRET_efficiency


def str2fp(line, id_str, res_bins, tagged_resn):
    nb_bins = len(res_bins) - 1
    fp_arr = ast.literal_eval(line.replace(id_str, ''))
    fp_list = [{resn: (np.histogram(fpc[resn], bins=res_bins)[0] > 0).astype(float) for resn in fpc} for fpc in fp_arr] # note: generates 99 bins, because 0-0.01 bin should not exist
    nb_snapshots = len(fp_list)
    fp_out = {}
    for resn in tagged_resn:
        fp_resn = np.sum([fp.get(resn, np.zeros(nb_bins, dtype=float)) for fp in fp_list], axis=0) / nb_snapshots
        fp_out[resn] = fp_resn
    return fp_out

def dist_str2fp(line, id_str, res_bins, tagged_resn):
    nb_bins = len(res_bins) - 1
    dist_fp_arr = ast.literal_eval(line.replace(id_str, ''))
    if type(dist_fp_arr[0]) == list: return {resn: np.zeros(nb_bins) for resn in tagged_resn}
    dist_fp_arr = [{resn: fp.get(resn, []) for resn in tagged_resn} for fp in dist_fp_arr]

    # probably not necessary for current implementation, but only keep fp's that are of correct length
    for resn in tagged_resn:
        fp_lens = [len(fp[resn]) for fp in dist_fp_arr]
        corr_len = max(set(fp_lens), key=fp_lens.count)
        dist_fp_arr = [fp for fp in dist_fp_arr if len(fp[resn]) == corr_len]

    # average distances per tag
    fp_arr = {}
    for resn in tagged_resn:
        fp_arr[resn] = [get_FRET_efficiency(d) for d in np.mean(np.vstack([fp[resn] for fp in dist_fp_arr]), axis=0)]

    fp_hist = {resn: np.clip((np.histogram(fp_arr[resn], bins=res_bins)[0] > 0).astype(float), 0, 1) for resn in fp_arr}  # Clip so fp is binary
    # fp_hist = {resn: (np.histogram(fp_arr[resn], bins=res_bins)[0] > 0).astype(float) for resn in fp_arr}
    return fp_hist

parser = argparse.ArgumentParser(description='Collect sort and store fingerprints from lattice model PDBs')
in_arg = parser.add_mutually_exclusive_group(required=True)
in_arg.add_argument('--in-dir', type=str, help='directory containing pdbs')
in_arg.add_argument('--lat-mod-dir', type=str, help='directory as produced by generate_lattice_models')
resolution_parser = parser.add_mutually_exclusive_group(required=True)
resolution_parser.add_argument('--efret-resolution', type=float,
                               help='FRET resolution in percentage points (integer!)')
resolution_parser.add_argument('--dist-resolution', type=float)
parser.add_argument('--original-dir', required=True,
                    help='directory containing starting npz files used in creating the lattice models')
parser.add_argument('--tagged-resn', required=True)
parser.add_argument('--lower-limit', type=float, default=0.2)
parser.add_argument('--out-pkl', type=str, required=True)

args = parser.parse_args()

_ = parse_output_dir(dirname(args.out_pkl), clean=False)
npz_list = parse_input_dir(args.original_dir, pattern='*.npz')
npz_dict = {basename(splitext(fn)[0]): fn for fn in npz_list}

pdb_parser = PDBParser()

if args.in_dir:
    pdb_list = parse_input_dir(args.in_dir, pattern='*.pdb')
else:
    pdb_id_list = [splitext(basename(f))[0] for f in os.scandir(f'{args.lat_mod_dir}/in_npz')]
    pdb_list = []
    for pdb_id in pdb_id_list:
        pdb_list.extend(parse_input_dir(args.lat_mod_dir, regex=f'{pdb_id}_[0-9]+.pdb'))

res, res_type = args.efret_resolution * 0.01, 'e_fret'
res_bins = np.arange(res, 1+res, res)

#
# if args.efret_resolution:
#     res, res_type = args.efret_resolution, 'e_fret'
#     res_bins = np.arange(0, 1+res, res)
# else:
#     res, res_type = args.dist_resolution, 'dist'

# Collect information on structures
struct_dict = dict()
for pdb_fn in pdb_list:
    pdb_id, struct_id = basename(splitext(pdb_fn)[0]).rsplit('_', 1)
    struct_id = int(struct_id)
    fingerprint_raw = []
    with open(pdb_fn, 'r') as fh:
        for line in fh.readlines():
            if 'REMARK   1 RG ' in line:
                rg = float(re.search('(?<=RG )[0-9\.]+', line).group(0))  # ensures you have the last Rg
            # if '1 FINGERPRINT' in line:
            #     fingerprint = str2fp(line, 'REMARK   1 FINGERPRINT ', res_bins, args.tagged_resn)
            #     break
            elif '1 DIST_FINGERPRINT' in line:
                fingerprint = dist_str2fp(line, 'REMARK   1 DIST_FINGERPRINT ', res_bins, args.tagged_resn)
                break
    if not np.sum([np.sum(fingerprint[resn]) for resn in fingerprint]):  # no tags visible means molecule was not observed
        continue

    # Find number of tagged residues
    npz_fn = npz_dict.get(pdb_id, None)
    if npz_fn is None:
        continue
    with np.load(npz_fn) as fh: resn_list = fh['sequence']
    seq_len = len(resn_list)
    nb_tags = np.sum(np.in1d(resn_list[1:], list(args.tagged_resn)))
    if not pdb_id in struct_dict:
        struct_dict[pdb_id] = {'fingerprints': dict(),
                               'fingerprints_raw': dict(),
                               'properties': {'number_of_tags': nb_tags,
                                              'rg_list': [],
                                              'sequence_length': seq_len}}
    struct_dict[pdb_id]['properties']['rg_list'].append(rg)
    struct_dict[pdb_id]['fingerprints'][struct_id] = fingerprint
    struct_dict[pdb_id]['fingerprints_raw'][struct_id] = fingerprint_raw
for pdb_id in struct_dict: struct_dict[pdb_id]['properties']['rg'] = np.mean(struct_dict[pdb_id]['properties']['rg_list'])

with open(args.out_pkl, 'bw') as fh: pickle.dump(struct_dict, fh)
