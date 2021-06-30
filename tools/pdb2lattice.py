import numpy as np
import os, sys
import argparse
from Bio.PDB import PDBParser
import pandas as pd
from itertools import combinations, chain
from math import sqrt
from time import sleep

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from helpers import get_neighbors, inNd, parse_input_dir, parse_output_dir, aa_dict_31, generate_pdb, get_cm, rotmats

ca_dist = 3.8 / sqrt(3)
cacb_dist = 1.53 / sqrt(3)
n1_dist = 1.48 / sqrt(3)
cacb_dist_unit = cacb_dist / ca_dist
n1_dist_unit = n1_dist / ca_dist

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


mirror_dims = list(combinations([0,1,2], 2)) + [tuple([i]) for i in range(3)] + [(0, 1, 2)]
mirror_dims_dual = list(combinations([0,1,2], 2))


def get_rotated_poses(coords):
    out_list = []
    for rot in rotmats:
        coords_rot = coords.copy()
        coords_rot[:, :3] = np.matmul(coords[:, :3], rot)
        out_list.append(coords_rot.copy())
    return out_list

def get_mirror_poses(coords):
    """
    Counts on mirroring in axes!
    """
    out_list = []
    for rot in rotmats:
        coords_rot = coords.copy()
        coords_rot[:, :3] = np.matmul(coords[:, :3], rot)
        out_list.append(coords_rot.copy())
        for md in mirror_dims:
            coords_mir = coords_rot.copy()
            coords_mir[:, md] = coords_mir[:, md] * -1
            out_list.append(coords_mir)
    return out_list


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

# helix_type_mod_dict = {
#     1: [np.array([[1, 1, 0],
#                   [0, 1, 1],
#                   [-1, 0, 0],
#                   [0, 0, -1]]),
#         np.array([[1, 1, 0],
#                   [0, 0, 1],
#                   [-1, 1, 0],
#                   [0, 0, -1]]),
#         np.array([[1, 1, 0],
#                   [0, 0, 1],
#                   [-1, 0, 0],
#                   [0, 1, -1]]),
#         np.array([[1, 0, 0],
#                   [0, 1, 1],
#                   [-1, 0, 0],
#                   [0, 1, -1]]),
#         np.array([[1, 0, 0],
#                   [0, 0, 1],
#                   [-1, 1, 0],
#                   [0, 1, -1]]),
#         ],
#     5: [np.array([[1, 0, 0],
#                  [-1, 0, 1],
#                  [0, 1, -1]]),
#         np.array([[1, 0, 1],
#                   [-1, 0, 0],
#                   [0, 1, -1]]),
#         np.array([[1, 0, 1],
#                   [-1, 1, 0],
#                   [0, 0, -1]]),
#         ]
# }

helix_type_mod_dict = {
    1: [np.array([[1, 1, 1],
                 [-1, 1, 1],
                 [-1, -1, 1],
                 [1, -1, 1]]),
        np.array([[1, 1, 1],
                  [-1, 1, 1],
                  [-1, -1, 1],
                  [1, -1, 1]]),
        np.array([[1, 1, 1],
                  [-1, 1, 1],
                  [-1, -1, 1],
                  [1, -1, 1]]),
        np.array([[1, 1, 1],
                  [-1, 1, 1],
                  [-1, -1, 1],
                  [1, -1, 1]])
        ],
    5: [np.array([[1, 1, 1],
                 [-1, 1, 1],
                 [-1, -1, 1],
                 [1, -1, 1]]),
        np.array([[1, 1, 1],
                  [-1, 1, 1],
                  [-1, -1, 1],
                  [1, -1, 1]]),
        np.array([[1, 1, 1],
                  [-1, 1, 1],
                  [-1, -1, 1],
                  [1, -1, 1]]),
        np.array([[1, 1, 1],
                  [-1, 1, 1],
                  [-1, -1, 1],
                  [1, -1, 1]])
        ]
}



# helix_type_mod_dict = {
#     1: [np.array([[1, 1, 1],
#                  [1, -1, 1],
#                  [-1, -1, 1],
#                  [-1, 1, 1]])],
#     5: [np.array([[1, 1, 1],
#                  [1, -1, 1],
#                  [-1, -1, 1],
#                  [-1, 1, 1]])],
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
# }

bcc_neighbor_mods = np.array([[1, 1, 1],
                              [-1, 1, 1],
                              [1, -1, 1],
                              [1, 1, -1],
                              [-1, -1, 1],
                              [-1, 1, -1],
                              [1, -1, -1],
                              [-1, -1, -1],
                              ])

def put_helix_on_lattice(tup):
    # todo: arteficially shortening helix REMOVE AFTERWARDS
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
        # coords_mirrored = get_rotated_poses(coords)
        coords_mirrored = get_mirror_poses(coords)
        coord_list.extend(coords_mirrored)
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

# def get_all_neighbors(c):
#     """
#     Gather diagonal and non-diagonal neighbors and concatenate for convenience
#     :param c:
#     :return:
#     """
#     neighbors = get_neighbors(c).astype(int)
#     diag_neighbors = get_diagonal_neighbors(c).astype(int)
#     neighbors = np.vstack((neighbors, diag_neighbors))
#     return neighbors



atm_names_bb = ['N', 'H', 'CA', 'HA', 'C', 'O']
atm_names_res = np.array(['B', 'G', 'D', 'E', 'F'])

parser = argparse.ArgumentParser(description='Translate fully atomistic pdb structures into lattice models with only CA,'
                                             'CB and N1 represented.')
parser.add_argument('--in-dir', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--cm-type', choices=['ca', 'bb_cm'], default='bb_cm')

args = parser.parse_args()

if args.cm_type == 'ca':
    atm_names_bb = ['CA']

pdb_list = parse_input_dir(args.in_dir, pattern='*.pdb')
out_dir = parse_output_dir(args.out_dir, clean=False)
npz_dir = parse_output_dir(out_dir + 'npz')
cm_dir = parse_output_dir(out_dir + 'cm')
pdb_lat_dir = parse_output_dir(out_dir + 'pdb_lat')

for pdb_fn in pdb_list:
    try:
        pdb_id = os.path.splitext(os.path.basename(pdb_fn))[0]
        # out_dir_cur = parse_output_dir(f'{out_dir}{pdb_id}')

        # 1. load structure
        p = PDBParser()
        pdb_structure = p.get_structure('structure', pdb_fn)
        mod = list(pdb_structure.get_models())[0]
        chain = [ch for ch in mod.get_chains()][0]

        # load secondary structure lines
        helix_txt_list = []
        sheet_txt_list = []
        sleep(0.5)
        with open(pdb_fn, 'r') as fh:
            for line in fh.readlines():
                if line.startswith('HELIX'): helix_txt_list.append(line)
                elif line.startswith('SHEET'): sheet_txt_list.append(line)

        # pre-allocate coordinate arrays
        ca_array, ca_array_lat = np.ones((len(chain), 3), float), np.ones((len(chain), 3), int)

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
                    # ca_array[ri, :3] = atm.get_coord()
                # elif np.any(np.in1d(list(atm.get_name()), atm_names_res)) \
                #         or res.resname == 'GLY' and atm.get_name() == '2HA': atms_res.append(atm)  # basically a placeholder
                if ri == 0 and atm.get_name() == 'N':
                    n_coord = atm.get_coord()
            ca_array[cidx] = get_cm(atms_bb)
            cidx += 1


        nb_res = len(resname_list)
        ca_array = ca_array[:nb_res, :]
        ca_array_lat = ca_array_lat[:nb_res, :]

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
        ca_array = ca_array / ca_dist # normalize to CA distance (lattice unit distances)
        n_coord = (n_coord - transl_zero) / ca_dist

        # pre-arrange beta sheets on lattice
        sheet_df = parse_sheet_info(sheet_txt_list)
        sheet_dict = {}
        for sheet_id, sdf in sheet_df.groupby('sheet_id'):
            res_idx_list, sheet_coords = put_sheet_on_lattice(sdf)
            sheet_dict[np.min(res_idx_list)] = (res_idx_list, sheet_coords)

        # pre-arrange alpha helices on lattice
        helix_df = parse_helix_info(helix_txt_list)
        helix_dict = {}
        for helix_id, tup in helix_df.iterrows():
            res_idx_list, helix_coords = put_helix_on_lattice(tup)
            res_idx_list = [resi2idx_dict[hidx] for hidx in res_idx_list if hidx in resi2idx_dict]
            # if not len(res_idx_list): continue
            helix_dict[np.min(res_idx_list)] = (res_idx_list, helix_coords)

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

        # # perform several passes over sheet structures to maximize number of contacts
        # for _ in range(20):
        #     for sheet_id, sdf in sheet_df.groupby('sheet_id'):
        #         strand_indices = [mi[0] for mi in sdf.index]
        #         lower_indices, higher_indices = strand_indices[:len(strand_indices) // 2][::-1], strand_indices[len(strand_indices) // 2:]
        #         all_indices = [(ind, 1) for ind in lower_indices] + [(ind, -1) for ind in higher_indices]
        #         for ind, lh in all_indices:
        #             cur_idx_list = np.arange(resi2idx_dict[sdf.loc[(ind, sheet_id)].resi_start],
        #                                      resi2idx_dict[sdf.loc[(ind, sheet_id)].resi_end])
        #             cur_tup = sdf.loc[(ind, sheet_id)]
        #             adjacent_tup = sdf.loc[(ind+lh, sheet_id)]
        #             adjacent_idx_list = np.arange(resi2idx_dict[adjacent_tup.resi_start],
        #                                           resi2idx_dict[adjacent_tup.resi_end])
        #             adjacent_strand_coords = ca_array_lat[adjacent_idx_list, :]
        #
        #             # gather neighbor coords of adjacent strand
        #             adj_neighbors = np.vstack([get_all_neighbors(ca_array_lat[c, :]) for c in adjacent_idx_list])
        #             adj_neighbors = adj_neighbors[np.invert(inNd(adj_neighbors[:, :3], ca_array_lat[:, :3])), :]
        #             for idx in cur_idx_list:
        #                 # Get all candidates adjacent to adjacent residues
        #                 prev_adj = get_all_neighbors(ca_array_lat[idx - 1, :])
        #                 next_adj = get_all_neighbors(ca_array_lat[idx + 1, :])
        #                 candidate_coords = prev_adj[inNd(prev_adj[:, :3], next_adj[:,:3]), :]
        #
        #                 # eliminate occupied coords and pick closest to adjacent strand
        #                 candidate_coords = candidate_coords[np.invert(inNd(candidate_coords[:, :3], ca_array_lat[:, :3])), :]
        #                 candidate_coords = np.vstack((candidate_coords, ca_array_lat[[idx], :]))
        #                 candidate_dists = [np.min(np.linalg.norm(adjacent_strand_coords - cc, axis=1)) for cc in candidate_coords]
        #                 ca_array_lat[idx, :] = candidate_coords[np.argmin(candidate_dists), :]

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

        # Get secondary structure df
        ss_df = pd.DataFrame(np.tile([0.05, 0.05, 0.90], (nb_res, 1)), columns=['H', 'S', 'L'], index=np.arange(nb_res))
        for _, tup in helix_df.iterrows():
            if tup.resi_end - tup.resi_start < 4: continue
            ah_idx = np.arange(resi2idx_dict[tup.resi_start], resi2idx_dict[tup.resi_end] - 3)
            ss_df.loc[ah_idx, :] = [0.90, 0.05, 0.05]
        for _, tup in sheet_df.iterrows():
            sh_idx = np.arange(resi2idx_dict[tup.resi_start], resi2idx_dict[tup.resi_end] + 1)
            ss_df.loc[sh_idx, :] = [0.05, 0.90, 0.05]
        ss_df = np.log(1 / ss_df - 1)

        # Save
        np.savez(f'{npz_dir}{pdb_id}_lat.npz',
                 coords=ca_array_lat[:, :3] * 2,
                 sequence=np.array([aa_dict_31[aa] for aa in resname_list]),
                 secondary_structure=ss_df,
                 )

        # iterate over CB coords, connect to CA
        # put_cb_on_lattice(cb_array, cb_mod_array, ca_array, ca_array_lat, resname_list, n_coord, n1_dist_unit)

        # put first N on lattice
        neighbors_unit = get_neighbors(ca_array_lat[0, :]).astype(int)
        neighbors = neighbors_unit * n1_dist_unit
        n_neigh_idx = np.argmin(np.linalg.norm(neighbors[:, :3] - n_coord, axis=1))
        n_coord_unit = neighbors_unit[n_neigh_idx, :]
        n_coord_lat = neighbors[n_neigh_idx, :]

        # Return from unit coords to angstrom
        ca_array_lat = ca_array_lat.astype(float)
        ca_array_lat[:, :3] = ca_array_lat[:, :3] * ca_dist
        n_coord_lat[:3] = n_coord_lat[:3] * ca_dist

        # Create pdb file
        pdb_txt = generate_pdb([np.squeeze(x) for x in np.vsplit(ca_array_lat[:, :3],len(ca_array_lat))], resname_list,
                               first_n=n_coord_lat)

        # Save
        # np.savez(f'{out_dir}{pdb_id}.npz', coords=ca_array_lat, aa_sequence=np.array([aa_dict_31[aa] for aa in resname_list]))
        with open(f'{pdb_lat_dir}{pdb_id}_lat.pdb', 'w') as fh:
            fh.write(pdb_txt)
    except Exception as e:
        print(f'Conversion failed for {pdb_id}: {e}')
        # raise
