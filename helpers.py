import numpy as np
import pandas as pd
from math import acos, atan2
import shutil
import pathlib
import os
import re
from glob import glob
from datetime import datetime
from random import random
import warnings
from Bio.PDB import PDBParser, HSExposureCB
from Bio.PDB.DSSP import DSSP
from itertools import groupby, chain, combinations
import operator
from copy import deepcopy
from collections import ChainMap
import traceback
import propka.run as pkrun

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
forster_radius = 54.0  # forster radius in Angstrom for Cy3-Cy5, according to Murphy et al. 2004


# --- IO ---
def parse_output_dir(out_dir, clean=False):
    out_dir = os.path.abspath(out_dir) + '/'
    if clean:
        shutil.rmtree(out_dir, ignore_errors=True)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_input_dir(in_dir, pattern=None, regex=None):
    if type(in_dir) != list: in_dir = [in_dir]
    out_list = []
    for ind in in_dir:
        if not os.path.exists(ind):
            raise ValueError(f'{ind} does not exist')
        if os.path.isdir(ind):
            ind = os.path.abspath(ind)
            if pattern is not None: out_list.extend(glob(f'{ind}/**/{pattern}', recursive=True))
            else: out_list.extend(glob(f'{ind}/**/*', recursive=True))
        else:
            if pattern is None: out_list.append(ind)
            elif pattern.strip('*') in ind: out_list.append(ind)
    if regex is not None:
        out_list = [fn for fn in out_list if re.search(regex, fn)]
    return out_list


def print_timestamp():
    return datetime.now().strftime('%y-%m-%d_%H:%M:%S')

def get_angle(v1, v2):
    # pv = np.cross(v1,v2)
    # pv = pv / np.linalg.norm(pv)
    #
    # pv2 = np.cross(v1, pv)
    # pv2 = pv2 / np.linalg.norm(pv2)
    #
    # # new basis
    # nb = np.vstack((v1, pv, pv2))
    #
    # v2_nb = np.dot(nb, v2)
    #
    # return np.degrees(atan2(np.linalg.det([v1,v2,pv]), np.dot(v1, v2)))
    return np.degrees(acos(max(-1, min(1, np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))))

def get_abs_dihedral(ca1, ca2, cb1, cb2):
    """
    Get absolute (!) dihedral from coordinates
    source: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    """
    p0 = cb1
    p1 = ca1
    p2 = ca2
    p3 = cb2

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.abs(np.degrees(np.arctan2(y, x)))


def get_compound_properties(pdb_id, pdb_fn, target_residue_list):
    """
    Retrieve selected properties from first chain (alphabetically) in a pdb file, in order:
    - compound name
    - synonyms
    - length of chain
    - list of number of residues present of given type(s)
    """
    pdb_parser = PDBParser()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pdb_obj = pdb_parser.get_structure(pdb_id, pdb_fn)
    chain = sorted(list(pdb_obj.get_chains()), key=lambda x: x.get_id())[0]
    chain_id = chain.get_id().lower()
    header = [ch for ch in pdb_obj.header['compound'].values() if chain_id in ch['chain']][0]
    aa_str = ''.join([aa_dict_31.get(res.get_resname(), '') for res in chain.get_residues()])
    res_occurances = [aa_str.count(res) for res in target_residue_list]
    return header['molecule'], header['synonym'], len(chain), res_occurances


def get_tagged_resi(mm, tagged_resn, aa_sequence, acc_resi):
    tagged_resi = {}
    tagged_resi_list = [acc_resi]  # acceptor-labeled resi
    for target_resn in tagged_resn:
        resi_list = []
        cur_mm = mm.get(target_resn, {target_resn: 1.0})
        for resi, resn in enumerate(aa_sequence):
            if resi not in tagged_resi_list and random() < cur_mm.get(resn, 0.0):
                resi_list.append(resi)
        tagged_resi[target_resn] = resi_list
        tagged_resi_list.extend(resi_list)
    tagged_resi['acc_tagged_resi'] = [acc_resi]
    return tagged_resi

# --- additional higher-dim numpy functions ---
def np2d_argmin(mat):
    d1 = mat.argmin()
    return d1 // mat.shape[1], d1 % mat.shape[1]


def inNd(a, b, assume_unique=False):
    if a.ndim == 1:
        a = np.expand_dims(a, 0)
    if b.ndim == 1:
        b = np.expand_dims(b, 0)
    if a.shape[1] != b.shape[1]:
        # return None
        raise ValueError(f'dim 1 of a ({a.shape[1]}) and b ({b.shape[1]}) do not match')
    # a = np.asarray(a, order='C')
    # b = np.asarray(b, order='C')
    a = a.ravel().view((np.void, a.dtype.itemsize*a.shape[1]))
    b = b.ravel().view((np.void, b.dtype.itemsize*b.shape[1]))
    return np.in1d(a, b, assume_unique)


def contains_double_coords(coords):
    sorted_coords = coords[np.lexsort(coords.T),:]
    double_bool = np.invert(np.any(np.diff(sorted_coords, axis=0), 1))
    return np.any(double_bool)


# --- lattice structure mutation functions ---
def pick_random_coords(coords):
    if coords.size == 0:
        raise ValueError('coordinates array is empty')
    elif coords.shape[0] == 1:
        return coords[0]
    else:
        return coords[np.random.randint(coords.shape[0]), :]


def get_neighbors(c, d=1):
    neighbors = np.tile(c, (6, 1))
    neighbors += np.row_stack((np.eye(3, dtype=int) * d, np.eye(3, dtype=int) * -1) * d)
    return neighbors

def get_diagonal_neighbors(c, d=1):
    neighbors = np.tile(c, (12, 1))
    m1 = np.abs(np.eye(3, dtype=int) - 1) * d
    m2 = np.array([[0, 1, -1],[1,0,-1],[1,-1,0]], dtype=int) * d
    m3 = m1 * -1
    m4 = m2 * -1
    mods = np.vstack((m1, m2, m3, m4))
    neighbors[:, :-1] += mods
    return neighbors

# def get_rotmat(dim):
#     th = 0.5 * np.pi * (1, -1)[dim < 0]
#     d = abs(dim)
#     if d == 1:
#         return np.array([[1, 0, 0, 0],
#                         [0, np.cos(th), -1*np.sin(th), 0],
#                         [0, np.sin(th), np.cos(th), 0],
#                         [0, 0, 0, 1]], dtype=int)
#     elif d == 2:
#         return np.array([[np.cos(th), 0, np.sin(th), 0],
#                         [0, 1, 0, 0],
#                         [-1*np.sin(th), 0, np.cos(th), 0],
#                         [0, 0, 0, 1]], dtype=int)
#     elif d == 3:
#         return np.array([[np.cos(th), -1*np.sin(th), 0, 0],
#                         [np.sin(th), np.cos(th), 0, 0],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1]], dtype=int)


# def get_transmat(c, dir):
#     em = np.eye(4, dtype=int)
#     em[:-1, 3] = c[:-1] * dir
#     return em


def pdb_coord(c):
    c_str_list = []
    for nc in c:
        c_str = f'{nc:.3f}'
        c_str = ' ' * (8 - len(c_str)) + c_str
        c_str_list.append(c_str)
    return ''.join(c_str_list)


def get_FRET_efficiency(dist):
    """
    return FRET efficiency for a given distance, between cy3-cy5
    """
    return 1.0/((dist / forster_radius) ** 6 + 1)

def get_FRET_distance(efret):
    return (efret ** -1 - 1) ** (1/6) * forster_radius


def get_pairs_mat(path):
    pairs_mat = pd.read_csv(path, sep='\s+', index_col=0, header=None)
    if len(pairs_mat.index[0]) == 3:
        pairs_mat.index = [aa_dict_31.get(str(aa).upper(), str(aa).upper()) for aa in pairs_mat.index]
    pairs_mat.columns = pairs_mat.index
    pairs_mat.sort_index(inplace=True)
    pairs_mat = pairs_mat.reindex(sorted(pairs_mat.columns), axis=1)
    pairs_mat.loc[:, 'TAG'] = 0.0
    pairs_mat.loc['TAG', :] = 0.0
    return pairs_mat


# --- residue name conversion dictionaries ---
aa_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    'X': 'TAG'
}

# For coloring html plots of structures:
# - blue: pos
# - red: neg
# - grey: polar
# - yellow: hydrophobic
aa_to_hp_dict = {
    'A': 'yellow',
    'C': 'grey',
    'D': 'red',
    'E': 'red',
    'F': 'yellow',
    'G': 'yellow',
    'H': 'blue',
    'I': 'yellow',
    'K': 'blue',
    'L': 'yellow',
    'M': 'yellow',
    'N': 'grey',
    'P': 'yellow',
    'Q': 'grey',
    'R': 'blue',
    'S': 'grey',
    'T': 'grey',
    'V': 'yellow',
    'W': 'yellow',
    'Y': 'grey'
}

aa_dict_31 = {v: k for k, v in aa_dict.items()}


atm_mass_dict = {
    'C': 12.0107,
    'O': 15.9994,
    'N': 14.0067,
    'S': 32.065
}

aa_mass_dict = {  # todo: where did I get these?? Does not quite match other sources...
    'A': 71.00779,
    'C': 103.1429,
    'D': 115.0874,
    'E': 129.11398,
    'F': 147.17386,
    'G': 57.05132,
    'H': 137.13928,
    'I': 113.15764,
    'K': 128.17228,
    'L': 113.15764,
    'M': 131.19606,
    'N': 114.10264,
    'P': 97.11518,
    'Q': 128.12922,
    'R': 156.18568,
    'S': 87.0773,
    'T': 101.10388,
    'V': 99.13106,
    'W': 186.2099,
    'Y': 163.17326
}

# cm_dist_df = pd.read_pickle(f'{__location__}/data/aa_pairwise_dist_df.pkl')

rotmat_dict = {
    1: np.array([[1,0,0],[0,0,-1],[0,1,0]]),
    -1: np.array([[1,0,0],[0,0,1],[0,1,0]]),
    2: np.array([[0,0,1],[0,1,0],[-1,0,0]]),
    -2: np.array([[0,0,1],[0,1,0],[1,0,0]]),
    3: np.array([[0,-1,0],[1,0,0],[0,0,1]]),
    -3: np.array([[0,1,0],[1,0,0],[0,0,1]])
}

# --- pair potentials functions ---

def remove_nonstd_residues(chain):
    """
    Remove non-standard residues that were in first/last position, check for nonstandard residues in other positions
    """
    res_list = list(chain.get_residues())
    caps = [res_list[0], res_list[-1]]
    while not all( [aa_dict_31.get(cap.get_resname(), False) for cap in caps]):
        [chain.detach_child(cap.get_id()) for cap in caps if not aa_dict_31.get(cap.get_resname(), False)]
        res_list = list(chain.get_residues())
        caps = [res_list[0], res_list[-1]]
    resbool_list = [ aa_dict_31.get(res.get_resname(), False) for res in  chain.get_residues()]
    return all(resbool_list)


def get_pairs_counts(pdb_fn, count_tup, parallel_bool, out_queue, disorder_only):
    try:
        pair_count_mat, ss_mat, count_dict = count_tup
        mat_update = np.eye(2, dtype=int)
        pdb_id = os.path.splitext(os.path.basename(pdb_fn))[0]
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_fn)
        model = structure[0]
        nb_residues = len(list(model.get_residues()))

        resn_dict = {res.id[1]: aa_dict_31[res.get_resname()] for res in model.get_residues()}

        # --- Secondary structure counts ---
        # todo: sheet df index changed!! No longer 0-based!
        helix_df, sheet_df = parse_ss_info(pdb_fn)
        helix_list, sheet_list = [], []
        if len(helix_df):
            helix_list = np.concatenate([np.arange(tup.resi_start, tup.resi_end+1) for _, tup in helix_df.iterrows()]) + 1
        if len(sheet_df):
            sheet_list = np.concatenate([np.arange(tup.resi_start, tup.resi_end+1) for _, tup in sheet_df.iterrows()]) + 1
        ss_dict = {res.id[1]: 'L' for res in model.get_residues()}
        for ssi in helix_list:
            if ssi in  ss_dict: ss_dict[ssi] = 'H'
        for ssi in sheet_list:
            if ssi in ss_dict: ss_dict[ssi] = 'S'
        for ssi in ss_dict:
            ss_mat.loc[resn_dict[ssi], ss_dict[ssi]] += 1

        # --- neighbor counts ---
        # hse = HSExposureCB(model, radius=6.6, offset=1)
        # hse

        # --- old stuff ---
        # # hse = HSExposureCB(model, radius=8, offset=1)
        # pdb_dict = {resi: {'resn': aa_dict_31[res.get_resname()], 'nb_adjacent': sum(hse[res.full_id[2:]][:2]),
        #                    'coords': [], 'ss': ss_dict.get(res.id[1], 'L')} for resi, res in enumerate(model.get_residues())}
        # cb_coords = np.empty((nb_residues, 3))
        # cb_coords[:,:] = np.nan
        #
        # # Collect c-beta coordinates and residue counts
        # for ri, res in enumerate(model.get_residues()):
        #     resname = aa_dict_31[res.get_resname()]
        #     assert pdb_dict[ri]['resn'] == resname
        #     if resname == 'G':
        #         cb = [atm for atm in res.get_atoms() if atm.get_name() == 'CA']
        #         coords = cb[0].get_coord()
        #         pdb_dict[ri]['coords'] = coords
        #     else:
        #         cb = [atm for atm in res.get_atoms() if atm.get_name() == 'CB']
        #         if not len(cb):
        #             if not out_queue: return pair_count_mat, count_dict
        #             out_queue.put((pair_count_mat, count_dict))
        #             return
        #         coords = cb[0].get_coord()
        #         cb_coords[ri, :] = coords
        #         pdb_dict[ri]['coords'] = coords
        #
        # # Update counts
        # for ri in pdb_dict:
        #     ss_mat.loc[pdb_dict[ri]['resn'], pdb_dict[ri]['ss']] += 1
        #     if disorder_only:
        #         if pdb_dict[ri]['ss'] != 'L': continue
        #     resname = pdb_dict[ri]['resn']
        #     # nb_neighbors = round(4 * (1 - pdb_dict[ri]['asa']))
        #     nb_neighbors = min(4, pdb_dict[ri]['nb_adjacent'])
        #     nb_water = 4 - nb_neighbors
        #     if nb_neighbors != 0:  # neighbor interactions
        #         exclude_idx = np.arange(max(0, ri-2), min(ri+2, nb_residues))
        #         coord_dist = np.linalg.norm(pdb_dict[ri]['coords'] - cb_coords, axis=1)
        #         proximity_order = np.argsort(coord_dist)
        #         proximity_order = proximity_order[np.invert(np.in1d(proximity_order, exclude_idx))]
        #         neighbors = [pdb_dict[ni] for ni in proximity_order[:nb_neighbors]]
        #         for neighbor in neighbors:
        #             if disorder_only:
        #                 if neighbor['ss'] != '-': continue
        #             pair_count_mat.loc[(resname, neighbor['resn']), (neighbor['resn'], resname)] += mat_update
        #     if nb_water != 0:  # water interactions
        #         count_dict['HOH'] += nb_water
        #         pair_count_mat.loc[(resname, 'HOH'), ('HOH', resname)] += mat_update * nb_water
        #     count_dict[resname] += 1
        if parallel_bool:
            out_queue.put((pair_count_mat, ss_mat, count_dict))
        else:
            return pair_count_mat, ss_mat, count_dict
    except:
        if parallel_bool:
            out_queue.put(count_tup)
        else:
            return count_tup


# --- to generate mock ss data from a pdb file, as if from ss prediction tool ---
def get_mock_ss_df(pdb_fn, resi2idx_dict, skip_outer_helix_indices=False):
    nb_res = len(resi2idx_dict)
    helix_txt_list = []
    sheet_txt_list = []
    with open(pdb_fn, 'r') as fh:
        for line in fh.readlines():
            if line.startswith('HELIX'):
                helix_txt_list.append(line)
            elif line.startswith('SHEET'):
                sheet_txt_list.append(line)
    helix_df = parse_helix_info(helix_txt_list)
    sheet_df = parse_sheet_info(sheet_txt_list)
    if skip_outer_helix_indices:
        helix_df = helix_df.query('length >=3').copy()
        helix_df.resi_start = helix_df.resi_start + 1
        helix_df.resi_end = helix_df.resi_end - 1
        helix_df.length = helix_df.length - 2

    helix_dict = {}
    for helix_id, tup in helix_df.iterrows():
        res_idx_list, helix_coords = put_helix_on_lattice(tup)
        res_idx_list = [resi2idx_dict[hidx] for hidx in res_idx_list if hidx in resi2idx_dict]
        helix_dict[np.min(res_idx_list)] = (res_idx_list, helix_coords)
    ss_df = pd.DataFrame(np.tile([0.05, 0.05, 0.90], (nb_res, 1)), columns=['H', 'S', 'L'], index=np.arange(nb_res))
    for _, tup in helix_df.iterrows():
        if tup.resi_end - tup.resi_start < 4: continue
        ah_idx = np.arange(resi2idx_dict[tup.resi_start], resi2idx_dict[tup.resi_end] - 3)
        ss_df.loc[ah_idx, :] = [0.90, 0.05, 0.05]
    for _, tup in sheet_df.iterrows():
        sh_idx = np.arange(resi2idx_dict[tup.resi_start], resi2idx_dict[tup.resi_end] + 1)
        ss_df.loc[sh_idx, :] = [0.05, 0.90, 0.05]
    ss_df = np.log(1 / ss_df - 1)
    return ss_df, helix_dict


def put_helix_on_lattice(tup):
    # generate all possible conformations for type of helix
    mods = helix_type_mod_dict[tup.type]
    nb_steps = mods[0].shape[0]
    coord_list = []
    for mod in mods:
        coords = np.zeros((tup.length, 3))
        for i in range(tup.length - 1):
            coords[i + 1, :3] = coords[i, :3] + mod[i % nb_steps, :]
        coords_mirrored = get_rotated_poses(coords)
        coord_list.extend(coords_mirrored)
    return np.arange(tup.resi_start, tup.resi_end + 1), coord_list


helix_type_mod_dict = {
    # 1: [np.array([[2, 2, 2],  # compressed
    #              [-2, 2, 0],
    #              [-2, -2, 2],
    #              [2, -2, 2]])],
    # 5: [np.array([[2, 2, 2],  # todo find better implementation for 3-10
    #              [-2, 2, 0],
    #              [-2, -2, 2],
    #              [2, -2, 2]])],
    1: [np.array([[2, -2, 2],  # uncompressed, full length
                 [2, 2, 2],
                 [-2, 2, 2],
                 [-2, -2, 2]])],
    5: [np.array([[2, -2, 2],
                 [2, 2, 2],
                 [-2, 2, 2],
                 [-2, -2, 2]])],

    # 5: [np.array([[1, 0, 0],
    #              [-1, 0, 1],
    #              [0, 1, -1]]),
    #     np.array([[1, 0, 1],
    #               [-1, 0, 0],
    #               [0, 1, -1]]),
    #     np.array([[1, 0, 1],
    #               [-1, 1, 0],
    #               [0, 0, -1]]),
    #     ]
}

def get_rotated_poses(coords):
    out_list = []
    for rot in rotmats:
        coords_rot = coords.copy()
        coords_rot[:, :3] = np.matmul(coords[:, :3], rot)
        out_list.append(coords_rot.copy())
    return out_list



rotmats_single = [
    np.array([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]]),
    np.array([[0, 0, 1],
              [0,1,0],
              [-1,0,0]]),
    np.array([[1,0,0],
              [0,0,-1],
              [0,1,0]]),
    np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]]),
    np.array([[-1,0,0],
              [0,1,0],
              [0,0,-1]]),
    np.array([[-1,0,0],
              [0,-1,0],
              [0,0,1]]),
    np.array([[1,0,0],
              [0,0,1],
              [0,-1,0]]),
    np.array([[0,0,-1],
              [0,1,0],
              [1,0,0]]),
    np.array([[0,1,0],
              [-1,0,0],
              [0,0,1]])
]

# mirror_dims = list(combinations([0,1,2], 2)) + [tuple([i]) for i in range(3)] + [(0, 1, 2)]
# mirror_dims_dual = list(combinations([0,1,2], 2))
rotmats_2 = [np.matmul(rotmats_single[md[0]], rotmats_single[md[1]]) for md in list(combinations(np.arange(len(rotmats_single)), 2))]
rotmats_3 = [np.matmul(np.matmul(rotmats_single[md[0]], rotmats_single[md[1]]), rotmats_single[md[2]]) for md in list(combinations(np.arange(len(rotmats_single)), 3))]
rotmats_w_doubles = list(chain.from_iterable([rotmats_single, rotmats_2, rotmats_3]))
rotmats = []
for rm in rotmats_w_doubles:
    dists = [np.linalg.norm(rm - el) for el in rotmats]
    if not len(dists) or 0 not in dists:
        rotmats.append(rm)


# ---

def parse_ss_info(pdb_fn):
    helix_txt_list = []
    sheet_txt_list = []
    with open(pdb_fn, 'r') as fh:
        for line in fh.readlines():
            if line.startswith('HELIX'):
                helix_txt_list.append(line)
            elif line.startswith('SHEET'):
                sheet_txt_list.append(line)
    helix_df = parse_helix_info(helix_txt_list)
    sheet_df = parse_sheet_info(sheet_txt_list)
    return helix_df, sheet_df


def parse_sheet_info(sheet_txt_list):
    """
    Parse list of strings of PDB SHEET records to pandas data frame with 0-based residue indices
    """
    sheet_df = pd.DataFrame(
        columns=['resi_start', 'resi_end', 'orientation', 'resi_hb_cur', 'resi_hb_prev'],
    index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=['strand_id', 'sheet_id']))
    for line in sheet_txt_list:
        ll = line.split()
        if ll[1] == '1':
            sheet_df.loc[(int(ll[1]),ll[2]), :] = [int(line[22:26]), int(line[33:37]), int(line[38:40]),
                                                   pd.NA, pd.NA]
        elif ll[1].isdigit():
            sheet_df.loc[(int(ll[1]),ll[2]), :] = [int(line[22:26]), int(line[33:37]), int(line[38:40]),
                                                   int(line[50:54]), int(line[65:69])]
        else:
            break
    return sheet_df

def parse_helix_info(helix_txt_list):
    """
    Parse list of strings of PDB HELIX records to a pandas data frame with 0-based residue indices
    """
    helix_df = pd.DataFrame(
        columns=['resi_start', 'resi_end', 'type', 'length']
    )
    for line in helix_txt_list:
        helix_df.loc[int(line[7:10]), :] = [int(line[21:25]), int(line[33:37]), int(line[39:40]), int(line[72:76])]
    return helix_df




def most_frequent_element(id_list):
    """
    Return most frequent list element, or with lowest index if tied
    from: https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    """
    id_list_sorted = sorted((idl, i) for i, idl in enumerate(id_list))
    groups = groupby(id_list_sorted, key=operator.itemgetter(0))
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(id_list)
        for _, where in iterable:
          count += 1
          min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    return max(groups, key=_auxfun)[0]


def generate_pdb(bb_coord_list, resnames, res_coord_list=None, first_n=None):
    # Create pdb file
    coord_txt_list = []
    conect_txt_list = []
    atm_nb = 1
    prev_ca = 1
    for ri, (bb_coord, resname) in enumerate(zip(bb_coord_list, resnames)):
        if ri == 0 and first_n is not None:
            # also register first N
            n_str = pdb_coord(first_n)
            coord_txt_list.append(
                f'HETATM{str(atm_nb).rjust(5)}  N   {resname} A{str(ri).rjust(4)}    {n_str}  1.00  1.00           N\n')
            atm_nb += 1
            conect_txt_list.append(f"CONECT{str(atm_nb - 1).rjust(5)}{str(atm_nb).rjust(5)}\n")

        ca_str = pdb_coord(bb_coord)
        coord_txt_list.append(
            f'HETATM{str(atm_nb).rjust(5)}  CA  {resname} A{str(ri).rjust(4)}    {ca_str}  1.00  1.00           C\n')
        conect_txt_list.append(
            f"CONECT{str(prev_ca).rjust(5)}{str(atm_nb).rjust(5)}\n")  # connect to previous CA (or first N)
        prev_ca = atm_nb
        atm_nb += 1
        if res_coord_list is not None and resname != 'GLY':
            cb_str = pdb_coord(res_coord_list[ri])
            coord_txt_list.append(
                f'HETATM{str(atm_nb).rjust(5)}  CB  {resname} A{str(ri).rjust(4)}    {cb_str}  1.00  1.00           C\n')
            atm_nb += 1
            conect_txt_list.append(f"CONECT{str(atm_nb - 2).rjust(5)}{str(atm_nb - 1).rjust(5)}\n")
    coord_txt_list.append(f'TER{str(atm_nb - 2).rjust(5)}      {resnames[-1]} A{str(len(resnames)).rjust(4)}\n')
    conect_txt = ''.join(conect_txt_list)
    coord_txt = ''.join(coord_txt_list)
    pdb_txt = coord_txt + conect_txt
    return pdb_txt

def get_cm(atm_list):
        return np.sum(np.row_stack([atm.get_coord() * atm.mass for atm in atm_list]), axis=0) / \
               np.sum([atm.mass for atm in atm_list])


def put_cb_on_lattice(cb_array, cb_mod_array, ca_array, ca_array_lat, resname_list, n_coord, n1_dist_unit):
    nb_res = len(resname_list)
    for ri in range(nb_res):
        if ri == 0:
            # also put first N on lattice
            neighbors_unit = get_neighbors(ca_array_lat[ri, :]).astype(int)
            neighbors = neighbors_unit * n1_dist_unit
            n_neigh_idx = np.argmin(np.linalg.norm(neighbors[:, :3] - n_coord, axis=1))
            n_coord_unit = neighbors_unit[n_neigh_idx, :]
            n_coord_lat = neighbors[n_neigh_idx, :]

        # apply same translation on CB as was done for CA
        if resname_list[ri] == 'GLY': continue
        transl = ca_array_lat[ri, :] - ca_array[ri, :]
        cb_coord = cb_array[ri, :] + transl

        # generate all possible positions for CB, pick closest to real coords
        neighbors = get_neighbors(ca_array_lat[ri, :]).astype(int)
        if ri == 0:
            adjacent_cas = np.vstack((ca_array_lat[ri+1, :], n_coord_unit))
        elif ri == nb_res - 1:
            adjacent_cas = np.expand_dims(ca_array_lat[ri-1, :], 0)
        else:
            adjacent_cas = np.vstack((ca_array_lat[ri-1, :], ca_array_lat[ri+1, :]))
        neighbors = neighbors[np.invert(inNd(neighbors[:, :3], adjacent_cas[:, :3])), :]
        cb_coord_overshot = neighbors[np.argmin(np.linalg.norm(neighbors[:, :3] - cb_coord[:3], axis=1)), :3]
        cb_mod_array[ri, :3] = cb_coord_overshot - ca_array_lat[ri, :3]

def list_sheet_series(ss_seq):
    dict_list = []
    in_sheet = False
    cur_list = []
    for ti, ss in enumerate(ss_seq):
        if ss == 'S':
            cur_list.append(ti)
        else:
            if len(cur_list): dict_list.append({cl: deepcopy(cur_list) for cl in cur_list})
            cur_list = []
    out_dict = dict(ChainMap(*dict_list))
    return out_dict


def find_df_start(data, reg_expression=r'Couple.*'):
    """Based on given regular expression finds the place where to start a dataframe."""

    r = re.compile(reg_expression)
    regmatch = np.vectorize(lambda x: bool(r.match(x)))
    result = np.where(regmatch(data[0].values) == True)
    return result[0][0]

def process_pka_file(pka_file, **kwargs):
    """Processes the output from propka. Outputs dataframe containing
    filtered residues based on given kwargs."""
    
    data = pd.read_fwf(pka_file, skiprows=52, widths=[
                    4, 3, 3, 7, 9, 7, 5, 7, 5, 7, 5,
                    3, 3, 7, 5, 3, 3, 7, 5, 3, 3 ], # propka output is a file with fixed width
                    header=None, skip_blank_lines=False)

    data = data.fillna('NaN')
    start = find_df_start(data, reg_expression=r'Co.*|-')
    stop = len(data.index)
    data = data.drop(index=(range(start, stop)), axis=0)
    data = data.replace('NaN', np.nan)
    data = data.dropna(how = 'all')
    data.insert(4, 'coupled_residues', data[3].str.contains('\*'))
    data[3] = data[3].str.rstrip('*')
    data[4] = data[4].str.rstrip('%')

    data = data.rename(
        columns =  {0:'residue_name',
                    1:'residue_id',
                    2:'chain_id',
                    3:'pKa',
                    4:'buried',
                    5:'desolvation_regular_1',
                    6:'desolvation_regular_2',
                    7:'effects_re_1',
                    8:'effects_re_2',
                    9:'sidechain_h_bond_1',
                    10:'sidechain_h_bond_2',
                    11:'sidechain_h_bond_3',
                    12:'sidechain_h_bond_4',
                    13:'backbone_h_bond_1',
                    14:'backbone_h_bond_2',
                    15:'backbone_h_bond_3',
                    16:'backbone_h_bond_4',
                    17:'coulombic_interaction_1',
                    18:'coulombic_interaction_2',
                    19:'coulombic_interaction_3',
                    20:'coulombic_interaction_4',
                    })

    data=data.astype({'pKa' : float})
    data=data.astype({'buried' : float})
    data.insert(4, 'reactivity', np.nan)

    filtered_residues = pd.DataFrame()
    for residue_name, parameters in kwargs.items():
        residues = data.loc[data['residue_name'] == residue_name]
        residues['reactivity']  = np.where( (residues.pKa >= parameters[0]) & (residues.pKa <= parameters[1])
                                         & (residues.buried <= parameters[3]), 'hiper-reactive', residues.reactivity)
        residues['reactivity']  = np.where( (residues.pKa >= parameters[0]) & (residues.pKa >= parameters[1])
                                         & (residues.buried <= parameters[3]), 'reactive', residues.reactivity)
        residues['reactivity']  = np.where( (residues.pKa >= parameters[2]) | (residues.buried > parameters[3]),
                                             'non-reactive', residues.reactivity)
        filtered_residues = pd.concat([filtered_residues, residues], axis=0)

    filtered_residues.dropna(inplace=True)
    filtered_residues = filtered_residues[['residue_name', 'residue_id',
                                            'reactivity', 'pKa',
                                            'buried']]
    return filtered_residues


def get_reactive_aa(pdb_file, residues_parameters=
                        {'CYS': [0, 7.65, 99, 85],
                       'LYS': [0, 9.75, 99, 85]},
                       rm_temp_files = True,
                       save=True):

    """Accepts PDB file and returns dictionary of residues of interest with pKa value
    and if the residue can be labeled or is it is an reactive residue.
    Parameters in  residues_parameters are default parameters
    that will be used to mark reactivity of the residue.
    Entry is defined as follows:
    residue name: [minimal_pKa, reactivity_threshold, maximal_pKa, maximal_burried_factor]."""
    


    try:
        pkrun.single(pdb_file, optargs=[ "--log-level=ERROR",])
        filename = pdb_file.partition('.')[0]
        reactive_aa = process_pka_file(filename +'.pka', **residues_parameters)
        if rm_temp_files == True:
            for f in glob(filename+ '.pka'):
                os.remove(f)
        if save == True:
            reactive_aa.to_csv(filename+'.csv', index=False)
        else:
            return reactive_aa
    except Exception as e:
        not_processed = open("not_processed.log", "a")
        not_processed.write(f"{pdb_file} {traceback.format_exc()}") #adds ID to log
        not_processed.close()
