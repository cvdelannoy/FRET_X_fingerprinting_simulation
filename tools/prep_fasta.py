import argparse, sys, os
import numpy as np
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from math import sqrt
from itertools import combinations
from os.path import basename, splitext

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from helpers import get_neighbors, inNd, parse_input_dir, parse_output_dir, aa_dict_31, generate_pdb, get_cm, get_diagonal_neighbors, put_cb_on_lattice, get_mock_ss_df, helix_type_mod_dict
import helpers as nhp
ss_type_dict = {'C': 'L',
                'E': 'S',
                'H': 'H'}


def get_pdb_coords(coords, aa_sequence, intermediate=False, conect_only=False):
    """
    Return coordinates in pdb format, as string
    :param intermediate: return without CONECT cards, required to create pdbs with multiple models
    :param conect_only: only return the CONECT cards
    :return:
    """
    n1_dist=1.48
    ca_dist = 3.8
    lat_dist = sqrt((0.5 * ca_dist) ** 2 / 3)

    coords_ca = coords - coords[0]  # translate to 0,0,0
    coords_ca = coords_ca * lat_dist  # unit distances to real distances

    cn = coords[1] * -1 * sqrt(n1_dist ** 2 / 3)  # stick on N1 in opposite direction of chain
    cn_str = nhp.pdb_coord(cn)
    # resn = nhp.aa_dict[aa_sequence[0]]
    resn = nhp.aa_dict.get(aa_sequence[0], aa_sequence[0])
    txt = f'HETATM    1  N   {resn} A   1    {cn_str}  1.00  1.00           N\n'

    # Add CA coordinates
    an = 2  # atom number, start at 2 for first N
    an_alpha = 1  # tracker of alpha carbon atom number, just for CONECT record
    resi = 1

    conect = ""
    for ci, ca in enumerate(coords_ca):
        # --- add alpha carbon CA ---
        # resn_str = nhp.aa_dict[aa_sequence[ci]]
        resn_str = nhp.aa_dict.get(aa_sequence[ci], aa_sequence[ci])
        resi_str = str(resi).rjust(4)
        ca_str = nhp.pdb_coord(ca)
        txt += f'HETATM{str(an).rjust(5)}  CA  {resn_str} A{resi_str}    {ca_str}  1.00  1.00           C\n'
        conect += f"CONECT{str(an_alpha).rjust(5)}{str(an).rjust(5)}\n"
        an_alpha = an
        an += 1
        resi += 1

    # Add terminus
    an_str = str(an).rjust(5)
    resn_str = nhp.aa_dict.get(aa_sequence[-1], aa_sequence[-1])
    resi_str = str(resi - 1).rjust(4)  # - 1: still on same residue as last CA
    txt += f'TER   {an_str}      {resn_str} A{resi_str}\n'
    if conect_only:
        return conect
    elif intermediate:
        return txt
    return txt + conect

def generate_lattice_helix(nb_res):
    mod = helix_type_mod_dict[1][0]
    nb_steps = mod.shape[0]
    coords = np.zeros((nb_res, 3))
    for i in range(nb_res - 1):
        coords[i + 1] = coords[i] + mod[i % nb_steps]
    return coords

parser = argparse.ArgumentParser(description='parse fasta and raptorX ss prediction into npz  file for'
                                             'lattice structure prediction')
parser.add_argument('--in-fasta', type=str, required=True)
parser.add_argument('--in-ss', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
args = parser.parse_args()

fasta_list = parse_input_dir(args.in_fasta, pattern='*.fasta')
ss_list = parse_input_dir(args.in_ss, pattern='*.ss3.txt')
out_dir = parse_output_dir(args.out_dir, clean=False)

ss_fn_dict = {splitext(splitext(basename(sfn))[0])[0]: sfn for sfn in ss_list}  # todo behavior when double extension?

for fasta_fn in fasta_list:
    sid = splitext(basename(fasta_fn))[0]
    if sid not in ss_fn_dict: continue
    ss_fn = ss_fn_dict[sid]
    with open(fasta_fn, 'r') as fh:
        seq = list(SimpleFastaParser(fh))[0][1]
    # with open(ss_fn, 'r') as fh:
    #     ss = fh.readlines()[-1].strip()
    # ss = ''.join([ss_type_dict[s] for s in ss])
    # assert len(ss) == len(seq)
    seqlen = len(seq)

    # Make secondary structure array
    ss_df = pd.read_csv(ss_fn, skiprows=2, names=['idx', 'resn', 'ss', 'H', 'S', 'L'], sep='\s+')
    ss = ss_df.ss.to_numpy()

    # Scale bonus by ss probability
    # ss_df = ss_df.loc[:, ('H', 'S', 'L')].copy()
    # ss_df[ss_df == 0.0] = 0.001
    # ss_df[ss_df == 1.0] = 0.999

    # Fix bonus at ~highest point, do not scale
    ss_df = pd.DataFrame(np.tile([0.05, 0.05, 0.90], (seqlen, 1)), columns=['H', 'S', 'L'], index=np.arange(seqlen))
    ss_df.loc[np.array(list(ss)) == 'H', :] = [0.90, 0.05, 0.05]
    ss_df.loc[np.array(list(ss)) == 'S', :] = [0.05, 0.90, 0.05]

    helix_indices, helix_on = [], False
    for si, css in enumerate(ss):
        if css == 'H':
            helix_indices.append(si)
        else:
            if len(helix_indices):
                ss_df.loc[helix_indices[-4:], :] = [0.05, 0.05, 0.90]
            helix_indices = []

    ss_df = np.log(1 / ss_df - 1)
    # ss_df.loc[:, 'L'] = 0
    # ss_df[ss_df > 0] = 0
    ss_array = ss_df.to_numpy()

    # make dict of helix structures
    helix_dict = {}
    helix_indices_treated = []
    for si, s in enumerate(ss):
        if si in helix_indices_treated: continue
        if s == 'H' and si != seqlen - 1:
            helix_idx_list = [si]
            for si2, s2 in enumerate(ss[si+1:]):
                if s2 == 'H':
                    helix_idx_list.append(si + si2 + 1)
                else:
                    break
            if len(helix_idx_list) < 4: continue
            helix_dict[si] = (helix_idx_list, generate_lattice_helix(len(helix_idx_list)))
            helix_indices_treated.extend(helix_idx_list)

    # Generate coords with prefolded alpha helices
    ca_array = np.zeros((len(seq), 3), int)
    prefold_indices = []
    for n in range(seqlen):
        if n in prefold_indices: continue
        if n != 0:
            mod = [-1, 1][n % 2]
            ca_array[n, :] = ca_array[n - 1, :] + np.array([2 * mod, 2 * mod, 2], int)
        if n in helix_dict:
            res_idx_list, pose = helix_dict[n]
            ca_array[res_idx_list, :] = pose + ca_array[n, :]
            prefold_indices.extend(res_idx_list)
    pdb_txt = get_pdb_coords(ca_array, seq)
    with open(f'{out_dir}{sid}_lat.pdb', 'w') as fh:
        fh.write(pdb_txt)
    np.savez(f'{out_dir}{sid}.npz', sequence=np.array(list(seq)), coords=ca_array, secondary_structure=ss_array)
