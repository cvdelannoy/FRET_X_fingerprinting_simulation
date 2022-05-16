import numpy as np
import os, sys, argparse, traceback
from Bio.PDB import PDBParser
import pandas as pd
import itertools
from itertools import permutations
from math import sqrt
from time import sleep

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from helpers import get_neighbors, inNd, parse_input_dir, parse_output_dir, aa_dict_31, generate_pdb, get_cm, rotmats, get_pairs_mat
from LatticeModelComparison import LatticeModelComparison

# --- Relevant lattice distances ---
ca_dist = 3.8 / sqrt(3)
cacb_dist = 1.53 / sqrt(3)
n1_dist = 1.48 / sqrt(3)
cacb_dist_unit = cacb_dist / ca_dist
n1_dist_unit = n1_dist / ca_dist

# --- Helix structure modifiers ---
bcc_neighbor_mods = m1 = np.array(list(set(permutations((1, 1, 1, -1, -1, -1), 3))))
m2_mods = np.array([[1,1,-1], [-1, 1, 1], [1,-1,1]])
m3p_mods = m2_mods * np.array([[-1,1,1], [1, -1, 1], [1,1,-1]])
m3n_mods = m2_mods * np.array([[1,-1,1], [1, 1, -1], [-1, 1, 1]])
m4p_mods = m3p_mods * m2_mods
m4n_mods = m3n_mods * m2_mods

helix_mods = list(itertools.chain.from_iterable([[
    np.vstack([mm1, mm1 * mm2, mm1 * mm3p, mm1 * mm4p]) if np.product(mm1) > 0
    else np.vstack([mm1, mm1 * mm2, mm1 * mm3n, mm1 * mm4n])
    for mm2, mm3p, mm3n, mm4p, mm4n in zip(m2_mods, m3p_mods, m3n_mods, m4p_mods, m4n_mods)] for mm1 in m1]))

helix_type_mod_dict = {
    1: helix_mods,
    5: helix_mods
}

# --- backbone and non-backbone atom names ---
atm_names_bb = ['N', 'H', 'CA', 'HA', 'C', 'O']
atm_names_res = np.array(['B', 'G', 'D', 'E', 'F'])


# --- functions ---
def parse_sheet_info(sheet_txt_list):
    """
    Parse list of strings of PDB SHEET records to pandas data frame
    """
    sheet_df = pd.DataFrame(
        columns=['resi_start', 'resi_end', 'orientation', 'resi_hb_cur', 'resi_hb_prev'],
        index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=['strand_id', 'sheet_id']))
    for line in sheet_txt_list:
        ll = line.split()
        if ll[1] == '1':
            sheet_df.loc[(int(ll[1]),ll[2]), :] = [int(line[22:26]), int(line[33:37]), int(line[38:40]),
                                                   pd.NA, pd.NA]
        else:
            sheet_df.loc[(int(ll[1]),ll[2]), :] = [int(line[22:26]), int(line[33:37]), int(line[38:40]),
                                                   int(line[50:54]), int(line[65:69])]
    return sheet_df


def parse_helix_info(helix_txt_list):
    """
    Parse list of strings of PDB HELIX records to a pandas data frame
    """
    helix_df = pd.DataFrame(
        columns=['resi_start', 'resi_end', 'type', 'length']
    )
    for line in helix_txt_list:
        resi_start, resi_end = int(line[21:25]), int(line[33:37])
        if resi_end - resi_start < 3: continue
        helix_df.loc[int(line[7:10]), :] = [int(line[21:25]), int(line[33:37]), int(line[39:40]), int(line[72:76])]
    return helix_df


def parse_ssbond_info(ssbond_txt_list, resi2idx_dict):
    ssbond_df = pd.DataFrame({'resi1': [resi2idx_dict[int(line[17:21])] for line in ssbond_txt_list],
                              'resi2': [resi2idx_dict[int(line[31:35])] for line in ssbond_txt_list]})
    ssbond_df.loc[:, 'max_resi'] = ssbond_df.max(axis=1)
    ssbond_df.loc[:, 'min_resi'] = ssbond_df.min(axis=1)
    ssbond_df.set_index('max_resi', inplace=True)
    return ssbond_df


def put_sheet_on_lattice(sheet_df):
    # todo output hbond indicator
    # strand_dict = {}
    idx_list = []
    coord_list = []
    for idx, tup in sheet_df.iterrows():
        len_strand = tup.resi_end - tup.resi_start + 1
        strand_array = np.zeros((len_strand, 3), dtype=int)
        strand_array[:, 1] = np.arange(len_strand)
        if idx[0] != 1:
            # Translate/rotate to correct position
            strand_array[:, 1] = strand_array[:, 1] * tup.orientation
            strand_array[:, 0] = idx[0]
            hb_idx = np.argwhere(tup.resi_hb_cur == np.arange(tup.resi_start, tup.resi_end + 1))[0, 0]
            hb_idx_prev = np.argwhere(tup.resi_hb_prev == prev_resi_range)[0, 0]
            strand_array[:, 1] = strand_array[:, 1] + (hb_idx_prev - hb_idx)
        prev_resi_range = np.arange(tup.resi_start, tup.resi_end + 1)
        idx_list.extend(list(range(tup.resi_start, tup.resi_end + 1)))
        coord_list.append(strand_array)
    coord_array = np.vstack(coord_list)
    return idx_list, coord_array


def put_helix_on_lattice(tup):
    tup.resi_start = tup.resi_start + 1
    tup.resi_end = tup.resi_end - 1
    tup.length = tup.length - 2
    # generate all possible conformations for type of helix
    mods = helix_type_mod_dict[tup.type]
    nb_steps = mods[0].shape[0]
    coord_list = []
    for mod in mods:
        coords = np.zeros((tup.length, 3))
        for i in range(tup.length - 1):
            coords[i + 1, :3] = coords[i, :3] + mod[i % nb_steps, :]
        coord_list.append(coords)
    return np.arange(tup.resi_start, tup.resi_end + 1), coord_list


def pick_best_pose(poses, first_lat_coord, ss_real_coords):
    best_pose = (None, np.inf)
    for pose in poses:
        pose = pose.copy()
        pose[:, :3] = pose[:, :3] + first_lat_coord[:3]
        sum_norm_diff = np.sum(np.linalg.norm(pose[:, :3] - ss_real_coords[:, :3], axis=1))
        if sum_norm_diff < best_pose[1]:
            best_pose = (pose, sum_norm_diff)
    return best_pose[0]


def get_all_neighbors(c):
    neighbors_out = bcc_neighbor_mods.copy()
    neighbors_out[:, :3] += c[:3]
    return neighbors_out


def parse_ss3_file(ss_fn):
    ss_df = pd.read_csv(ss_fn, skiprows=2, names=['idx', 'resn', 'ss', 'H', 'S', 'L'], sep='\s+')
    ss = ss_df.ss.to_numpy()
    seqlen = len(ss)
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

    # construct helix_df
    helix_list = []
    helix_bool = False
    helix_start = None
    for si, s in enumerate(ss):
        if s == 'H' and not helix_bool:
            helix_start, helix_bool = si, True
        if s != 'H' and helix_bool:
            helix_length = si - helix_start
            if helix_length >= 5:
                helix_list.append(pd.Series({'resi_start': helix_start, 'resi_end': si - 1,
                                             'type': 1, 'length': helix_length}))
            helix_bool = False
    helix_df = pd.concat(helix_list, axis=1).T

    return ss_df, helix_df


parser = argparse.ArgumentParser(description='Translate fully atomistic pdb structures into lattice models with only CA,'
                                             'CB and N1 represented.')
parser.add_argument('--in-dir', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--cm-type', choices=['ca', 'bb_cm'], default='ca')
parser.add_argument('--ss3', type=str,
                    help='Use ss3.txt file for secondary structure instead of HELIX and SHEET cards in pdb.')
parser.add_argument('--acc-tagged-resi', type=int, default=0,
                    help='Define index of residue to which acceptor docker strand should be attached, -1 for C-terminus [default: 0]')
parser.add_argument('--refinement-rounds', type=int, nargs=2, default=[1000, 1000],
                    help='Refine structure in two rounds of RMSD-minimization on lattice [default: 1000 1000]')
parser.add_argument('--disordered-terminal', type=str, required=False,
                    help='Introduce disorder from a given resi, model as random walk. Must be formatted as '
                         '[0-9]+: or :[0-9]+ [default: None].')

args = parser.parse_args()

if args.cm_type == 'ca':
    atm_names_bb = ['CA']

pdb_list = parse_input_dir(args.in_dir, pattern='*.pdb')
out_dir = parse_output_dir(args.out_dir, clean=False)
npz_dir = parse_output_dir(out_dir + 'npz')
cm_dir = parse_output_dir(out_dir + 'cm')
pdb_lat_dir = parse_output_dir(out_dir + 'pdb_lat')

error_log = out_dir + 'errors.log'
with open(error_log, 'w') as fh: fh.write('id\terror_message\n')

for pdb_fn in pdb_list:
    try:
        pdb_id = os.path.splitext(os.path.basename(pdb_fn))[0]

        # load structure
        p = PDBParser()
        pdb_structure = p.get_structure('structure', pdb_fn)
        mod = list(pdb_structure.get_models())[0]
        chain = [ch for ch in mod.get_chains()][0]

        # load secondary structure lines
        helix_txt_list = []
        sheet_txt_list = []
        ssbond_txt_list = []
        sleep(0.5)
        with open(pdb_fn, 'r') as fh:
            for line in fh.readlines():
                if line.startswith('HELIX'): helix_txt_list.append(line)
                elif line.startswith('SHEET'): sheet_txt_list.append(line)
                elif line.startswith('SSBOND'): ssbond_txt_list.append(line)

        # pre-allocate coordinate arrays
        ca_array = np.ones((len(chain), 3), float)

        # Gather resn's, coordinates
        resname_list = []
        resi2idx_dict = {}
        cidx = 0
        for ri, res in enumerate(chain.get_residues()):
            if res.resname not in aa_dict_31: continue
            resname_list.append(res.resname)
            resi2idx_dict[res.id[1]] = cidx
            atms_bb = []
            atms_res = []
            atm_list = list(res.get_atoms())
            atm_names_cur = [atm.get_name() for atm in atm_list]
            if 'CA' not in atm_names_cur: continue  # ensure that at least the CA is present
            for atm_name, atm in zip(atm_names_cur, atm_list):
                if atm_name in atm_names_bb: atms_bb.append(atm)
                if ri == 0 and atm.get_name() == 'N':
                    n_coord = atm.get_coord()
            ca_array[cidx] = get_cm(atms_bb)
            cidx += 1

        nb_res = len(resname_list)
        ca_array = ca_array[:nb_res, :]

        if np.any([resn not in aa_dict_31 for resn in resname_list]):
            print(f'Not all residues are standard in {pdb_id}, skipping...')
            continue

        # Save cm version of structure
        pdb_txt = generate_pdb([np.squeeze(x) for x in np.vsplit(ca_array, len(ca_array))], resname_list,
                                   first_n=n_coord)
        sleep(0.5)
        with open(f'{cm_dir}{pdb_id}_cm.pdb', 'w') as fh: fh.write(pdb_txt)

        # translate to unit lattice
        transl_zero = ca_array[0, :].copy()
        ca_array = ca_array - transl_zero  # translate to start at 0 0 0
        ca_array = ca_array / ca_dist  # normalize to CA distance (lattice unit distances)
        n_coord = (n_coord - transl_zero) / ca_dist

        # Get secondary structure data
        sheet_df = parse_sheet_info(sheet_txt_list)
        helix_df = parse_helix_info(helix_txt_list)
        if args.ss3:
            ss_df, helix_df = parse_ss3_file(args.ss3)

        else:
            ss_df = pd.DataFrame(np.tile([0.05, 0.05, 0.90], (nb_res, 1)), columns=['H', 'S', 'L'],
                                 index=np.arange(nb_res))
            for _, tup in helix_df.iterrows():
                if tup.resi_end - tup.resi_start < 4: continue
                ah_idx = np.arange(resi2idx_dict[tup.resi_start], resi2idx_dict[tup.resi_end] - 3)
                ss_df.loc[ah_idx, :] = [0.90, 0.05, 0.05]
            for _, tup in sheet_df.iterrows():
                sh_idx = np.arange(resi2idx_dict[tup.resi_start], resi2idx_dict[tup.resi_end] + 1)
                ss_df.loc[sh_idx, :] = [0.05, 0.90, 0.05]
            ss_df = np.log(1 / ss_df - 1)

        # pre-arrange beta sheets on lattice
        sheet_dict = {}
        for sheet_id, sdf in sheet_df.groupby('sheet_id'):
            res_idx_list, sheet_coords = put_sheet_on_lattice(sdf)
            sheet_dict[np.min(res_idx_list)] = (res_idx_list, sheet_coords)

        # pre-arrange alpha helices on lattice
        helix_dict = {}
        for helix_id, tup in helix_df.iterrows():
            res_idx_list, helix_coords = put_helix_on_lattice(tup)
            res_idx_list = [resi2idx_dict[hidx] for hidx in res_idx_list if hidx in resi2idx_dict]
            # if not len(res_idx_list): continue
            helix_dict[np.min(res_idx_list)] = (res_idx_list, helix_coords)

        # parse ss-bond lines
        ssbond_df = parse_ssbond_info(ssbond_txt_list, resi2idx_dict)

        # pre-allocate lat coordinate array
        ca_array_lat = np.zeros((nb_res, 3), int)

        # iterate over CA coords and put on lattice
        preset_idx = []
        for ri in range(1, nb_res):
            if ri in preset_idx:  # if coord has been set previously as part of a secondary structure, do not set again
                continue
            # generate all possible positions for new CA, pick closest to real coords
            neighbors = get_all_neighbors(ca_array_lat[ri - 1, :])
            neighbors = neighbors[np.invert(inNd(neighbors[:, :3], ca_array_lat[:, :3])), :]
            ca_array_lat[ri, :] = neighbors[np.argmin(np.linalg.norm(neighbors[:, :3] - ca_array[ri, :3], axis=1)), :]
            if ri in helix_dict:
                res_idx_list, poses = helix_dict[ri]
                helix_coords_original = ca_array[res_idx_list, :3]
                best_pose = pick_best_pose(poses, ca_array_lat[ri, :], helix_coords_original)
                ca_array_lat[res_idx_list, :] = best_pose
                preset_idx.extend(res_idx_list)

        # Attempt to correct any overlapping coordinates
        unique_rows, counts = np.unique(ca_array_lat, return_counts=True, axis=0)
        for ri, cnt in enumerate(counts):
            if cnt > 1:
                cnt_resolved = 0
                nur = unique_rows[ri]
                nur_idx_array = np.argwhere(np.all(ca_array_lat == nur, axis=1)).squeeze(-1)
                for nur_idx in nur_idx_array:
                    if nur_idx == 0:
                        candidates = get_all_neighbors(ca_array_lat[nur_idx+1])
                    elif nur_idx == len(ca_array_lat)-1:
                        candidates = get_all_neighbors(ca_array_lat[nur_idx - 1])
                    else:
                        nur_coords = ca_array_lat[nur_idx - 1:nur_idx + 2, :]
                        candidates = get_all_neighbors(ca_array_lat[nur_idx-1, :])
                        candidates = candidates[inNd(candidates, get_all_neighbors(nur_coords[2]))]
                    candidates = candidates[np.invert(inNd(candidates, ca_array_lat))]
                    if not len(candidates): continue
                    candidate = candidates[np.random.randint(len(candidates))]
                    ca_array_lat[nur_idx] = candidate
                    cnt_resolved += 1
                    if cnt_resolved == cnt-1: break

        ca_array_lat = ca_array_lat * 2  # double, as only even coords are valid



        # Refine structure
        aa_seq = np.array([aa_dict_31[aa] for aa in resname_list])
        lmc1 = LatticeModelComparison(mod_id=0, lattice_type='bcc', beta=0.5,
                                      nb_steps=3, store_rg=False,
                                      starting_structure=ca_array_lat,
                                      coords=ca_array_lat,
                                      aa_sequence=aa_seq, tagged_resi=dict(), secondary_structure=ss_df,
                                      no_regularization=True,
                                      pairs_mat=get_pairs_mat(f'{__location__}/../potential_matrices/aa_water2_abeln2011.txt'),
                                      ssbond_df=ssbond_df,
                                      cm_coords=ca_array, finetune_structure=True)
        lmc1.do_mc(args.refinement_rounds[0], silent=False)

        lmc2 = LatticeModelComparison(mod_id=0, lattice_type='bcc', beta=np.nan,
                                      nb_steps=1, store_rg=False,
                                      starting_structure=ca_array_lat,
                                      coords=lmc1.best_model.coords,
                                      aa_sequence=aa_seq, tagged_resi=dict(), secondary_structure=ss_df,
                                      no_regularization=True,
                                      pairs_mat=get_pairs_mat(
                                          f'{__location__}/../potential_matrices/aa_water2_abeln2011.txt'),
                                      ssbond_df=ssbond_df,
                                      cm_coords=ca_array, finetune_structure=True)
        lmc2.do_mc(args.refinement_rounds[1], silent=False)
        ca_array_lat = lmc2.best_model.coords

        # Replace part of structure with random walk, if required
        if args.disordered_terminal:
            s2_neighbors = bcc_neighbor_mods * 2
            if args.disordered_terminal[0] == ':':
                disorder_idx = np.arange(0,int(args.disordered_terminal[1:]))[::-1]
                last_coord = ca_array_lat[disorder_idx[0]+1]
            elif args.disordered_terminal[-1] == ':':
                disorder_idx = np.arange(int(args.disordered_terminal[:-1]), nb_res)
                last_coord = ca_array_lat[disorder_idx[0] - 1]
            ca_array_lat[disorder_idx, :] = 1
            for di in disorder_idx:
                coord_candidates = s2_neighbors + last_coord
                coord_candidates = coord_candidates[np.invert(inNd(coord_candidates, ca_array_lat))]
                if not len(coord_candidates):
                    raise ValueError('Could not finish disordered terminal task')
                ca_array_lat[di] = coord_candidates[np.random.randint(len(coord_candidates))]
                last_coord = ca_array_lat[di]
            ss_df.loc[disorder_idx, :] = 0


        # Save
        atr = args.acc_tagged_resi if args.acc_tagged_resi != -1 else len(ca_array_lat) - 1
        np.savez(f'{npz_dir}{pdb_id}_lat.npz',
                 coords=ca_array_lat[:, :3],
                 sequence=np.array([aa_dict_31[aa] for aa in resname_list]),
                 secondary_structure=ss_df,
                 acc_tagged_resi=atr
                 )

        # put first N on lattice
        neighbors_unit = get_neighbors(ca_array_lat[0, :]).astype(int)
        neighbors = neighbors_unit * n1_dist_unit
        n_neigh_idx = np.argmin(np.linalg.norm(neighbors[:, :3] - n_coord, axis=1))
        n_coord_unit = neighbors_unit[n_neigh_idx, :]
        n_coord_lat = neighbors[n_neigh_idx, :]

        # Return from unit coords to angstrom
        ca_array_lat = ca_array_lat.astype(float)
        ca_array_lat[:, :3] = ca_array_lat[:, :3] / 2 * ca_dist
        n_coord_lat[:3] = n_coord_lat[:3] * ca_dist

        # Create pdb file
        pdb_txt = generate_pdb([np.squeeze(x) for x in np.vsplit(ca_array_lat[:, :3],len(ca_array_lat))], resname_list,
                               first_n=n_coord_lat)

        # Save
        with open(f'{pdb_lat_dir}{pdb_id}_lat.pdb', 'w') as fh:
            fh.write(pdb_txt)
    except Exception as e:
        print(f'Conversion failed for {pdb_id}: line {e.__traceback__.tb_lineno}, {e}')
        traceback.print_exc()
        with open(error_log, 'a') as fh: fh.write(f'{pdb_id}\t{e}\n')
