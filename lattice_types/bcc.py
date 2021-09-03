import os, sys
from math import sqrt, copysign
import pandas as pd
import numpy as np
import helpers as nhp
from helpers import rotmat_dict, rotmats
from LatticeModel import LatticeModel
from cached_property import cached_property
import random
from itertools import combinations
import plotly as py
import plotly.graph_objs as go
from Bio.PDB import PDBParser
from Bio.PDB.QCPSuperimposer import QCPSuperimposer
pdb_parser = PDBParser()
imposer = QCPSuperimposer()


neighbor_mods = np.array([
    [2, 2, 2],[-2, -2, -2],
    [-2, 2, 2],[2, -2, 2],[2, 2, -2],
    [-2, -2, 2],[-2, 2, -2],[2, -2, -2]
])

cubic_neighbor_mods = np.array([
    [0,0,4], [0,4,0], [4,0,0],
    [0,0,-4],[0,-4,0], [-4,0,0],
])

neighbor_mods_d2 = np.unique(np.vstack([nm + neighbor_mods for nm in neighbor_mods]), axis=0)
neighbor_mods2 = np.vstack((neighbor_mods, neighbor_mods_d2))
mod2mod_dict = {nmi: np.argwhere(nhp.inNd(neighbor_mods2, nm1 * 2))[0,0] for nmi, nm1 in  enumerate(neighbor_mods)}

tag_mods_single = [np.cumsum(np.tile(mod, (10,1)), axis=0) for mod in neighbor_mods]
# test: allow cubic paths for tags
cubic_tag_mods_single = [np.cumsum(np.tile(mod, (10,1)), axis=0) for mod in cubic_neighbor_mods]
tag_mods_single.extend(cubic_tag_mods_single)
tag_mods_bulk = []
for tm in tag_mods_single:
    tmb = np.unique(np.vstack([tms + neighbor_mods2 for tmi, tms in enumerate(tm) if tmi > 1]), axis=0)
    tmb = tmb[np.invert(nhp.inNd(tmb, tm))]
    tag_mods_bulk.append(tmb)
tag_mods = list(zip(tag_mods_single, tag_mods_bulk))

quad_neighbor_mods_abs = np.array([
    [0, 0, 4],
    [0, 4, 0],
    [4, 0, 0]
])

helix_array = np.array([[0, 0, 0],
                        [2, -2, 2],
                        [4, 0, 4],
                        [2, 2, 6],
                        [0, 0, 8]])

rotated_helix_array_list = [np.matmul(helix_array, rot) for rot in rotmats]

# mirror_dims = list(combinations([0,1,2], 2)) + [tuple([i]) for i in range(3)] + [(0, 1, 2)]
# mirrored_rotated_helix_array_list = [rhm for rhm in rotated_helix_array_list]

# helix_mod = np.array([[2, -2, 2],
#                       [2, 2, 2],
#                       [-2, 2, 2],
#                       [-2, -2, 2]])

# helix with equidistant neighbors
helix_v_truth = np.array([[6, 2, -2, 2],
                          [-6, -2, 2, -2]])
helix_h_truth = np.array([0, 0, 8])

# helix with 1 quad face transition
# helix_h_truth = np.array([0, 0, 6])
#
# helix_v_truth = np.array([[6, 0, -2, 2],
#                           [-6, 0, 2, -2]])

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    from: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class Lattice(LatticeModel):
    """Class containing all that pertains to a particular type of lattice (initialization, allowed moves etc.)
        lattice type: body-centered cubic (bcc)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pdb_id = kwargs.get('pdb_id', 'unknown')
        self.experimental_mode = kwargs['experimental_mode']
        self.no_regularization = kwargs.get('no_regularization', False)
        self.ca_dist = 3.8  # actual CA distance
        self.lat_dist = sqrt((0.5 * self.ca_dist) ** 2 / 3)  # distance of lattice edge
        self.linker_dist = 21  # Distance tagged CA to dye
        self.linker_dist_lat = sqrt(self.linker_dist ** 2 / 3)
        self.n1_dist = 1.48  # estimate of distance N to CA
        self.pairs_mat = kwargs['pairs_mat']
        self.ss_df = kwargs['secondary_structure']
        # self.sheet_series = nhp.list_sheet_series(self.ss_sequence)
        self.coords = kwargs.get('coords', None)
        self.prev_coords = self.coords.copy()
        self.branch_rotation_idx_list = list(range(len(rotmats)))

        self.cm_coords = kwargs.get('cm_coords', None)
        self.finetune_structure = kwargs.get('finetune_structure', False)

    def coords_are_valid(self):
        """
        For testing purposes!
        """
        for i, c in enumerate(self.coords[:-1]):
            if not np.all(np.abs(self.coords[i+1] - c) == 2): return False
        return True

    @property
    def cm_coords(self):
        return self._cm_coords

    @cm_coords.setter
    def cm_coords(self, coords):
        """
        Translate cm coords to unit lattice
        """
        if coords is None:
            self._cm_coords = None
            return
        self._cm_coords = (coords - coords[0]) / self.lat_dist

    @cached_property
    def sheet_block_dict(self):
        out_dict = {}
        cur_block_idx = 0
        in_block = False
        for si, ss in enumerate(self.ss_sequence):
            if ss == 'S':
                if not in_block:
                    cur_block_idx += 1
                    in_block = True
                out_dict[si] = cur_block_idx
            else:
                in_block = False
        return out_dict

    @property
    def ss_df(self):
        return self._ss_df

    @ss_df.setter
    def ss_df(self, df):
        df.loc[self.tagged_resi, :] = 00,   4,   4
        df.loc[:, 'L'] = 0
        df[df > 0] = 0
        self._ss_df = df

    # --- mutations ---

    def apply_n_steps(self, n):
        global_fun_list = [
            # self.apply_crankshaft_move,
            self.apply_branch_rotation,
            self.apply_corner_flip,
            # self.apply_pull_move  # screws up helices, can't get it right
        ]

        for _ in range(n):
            random.shuffle(global_fun_list)
            if global_fun_list[0](): pass
            elif global_fun_list[1](): pass
            # elif global_fun_list[2](): pass
            # elif global_fun_list[3](): pass
            else: return False
            self.set_hash_list()
            self.__dict__.pop('e_matrix', None)

        return True

    def check_helicity(self):
        # test: see if helices are still in place
        for ci, ss in self.ss_df.iterrows():
            if ss.H >= 0: continue
            helix_candidate = self.coords[ci:ci + 5] - self.coords[ci]
            hel_dists = [np.linalg.norm(helix_candidate - hel) for hel in rotated_helix_array_list]
            if not np.any(np.array(hel_dists) == 0):
                return ci

    @property
    def branch_rotation_idx_list(self):
        random.shuffle(self._branch_rotation_idx_list)
        return self._branch_rotation_idx_list

    @branch_rotation_idx_list.setter
    def branch_rotation_idx_list(self, bri_list):
        self._branch_rotation_idx_list = bri_list

    def apply_branch_rotation(self):
        mutations = list(range(-3, 4))
        mutations.remove(0)
        random.shuffle(mutations)
        idx_list = list(range(self.seq_length - 1))
        idx_list = np.array(idx_list)[self.ss_sequence[:-1] != 'H']
        random.shuffle(idx_list)  # randomize positions to check
        for ci in idx_list:  # omit last position, where rotation does not make sense
            for mi in self.branch_rotation_idx_list:
                candidate = self.branch_rotation(self._coords[ci + 1:, :], self._coords[ci, :], mi)
                if not np.any(nhp.inNd(candidate, self.coords[:ci, :])):
                    self._coords[ci + 1:, :] = candidate
                    return True
                # candidate[ci + 1:, :] = self.branch_rotation(self._coords[ci + 1:, :], self._coords[ci, :], mut)
                # if self.is_valid_candidate(candidate):
                #     self.coords = candidate
                #     return True
        return False

    def apply_pull_move(self):
        direction = [-1, 1]
        random.shuffle(direction)
        idx_list = list(range(2, self.seq_length - 2))
        idx_list = np.array(idx_list)[self.ss_sequence[2:-2] != 'H']
        random.shuffle(idx_list)  # randomize positions to check
        candidate_found = False
        for ri in idx_list:
            for dir in direction:
                if self.ss_sequence[ri + dir] == 'H' or self.ss_sequence[ri + dir * 2] == 'H': continue
                # Candidates for first moved atom should be
                l0_candidates = self.coords[ri + dir] + neighbor_mods_d2  # reachable from their old pos by 2 steps
                l0_candidates = l0_candidates[nhp.inNd(l0_candidates, self.coords[ri] + neighbor_mods)]  # adjacent to non-moved atom
                l0_candidates = l0_candidates[np.invert(nhp.inNd(l0_candidates, self.coords))] # unoccupied
                if not len(l0_candidates): continue
                np.random.shuffle(l0_candidates)
                for l0 in l0_candidates:
                    # Candidates for second moved atom should be...
                    l1_candidates = self.coords[ri + dir * 2] + neighbor_mods_d2  # reachable from their old pos by 2 steps
                    l1_candidates = l1_candidates[nhp.inNd(l1_candidates, l0 + neighbor_mods)]  # adjacent to new l0 coord
                    if not len(l1_candidates): continue
                    l1_candidates = l1_candidates[np.invert(self.inNd(l1_candidates))]  # unoccupied
                    if not len(l1_candidates): continue
                    l0_idx = ri + dir
                    d2_pos = l1_candidates[np.random.randint(len(l1_candidates))]

                    # Get position for third moved atom: between new d2 position and old d2 position
                    d1_candidates = d2_pos + neighbor_mods
                    d1_pos = d1_candidates[nhp.inNd(d1_candidates, self.coords[ri + dir * 2] + neighbor_mods)][0]
                    if self.inNd(d1_pos)[0]: continue
                    self._coords[ri + dir] = l0
                    change_idx = np.arange(ri + 2, self.seq_length) if dir == 1 else np.arange(ri-1)[::-1]
                    candidate_found = True
                    break
                if candidate_found: break
            if candidate_found: break
        if not candidate_found:
            return False

        # Fill in positions
        prev_c = l0_idx
        first_H = True
        for c in change_idx:
            if self.ss_sequence[c] != 'H' and np.all(np.abs(self.coords[c] - self.coords[prev_c]) == 2):
                break
            if self.ss_sequence[c] == 'H':
                if first_H:
                    helix_transl = self.coords[c] - d2_pos
                    self.coords[c] = d2_pos
                    first_H = False
                else:
                    d2_pos = d1_pos
                    d1_pos = self.coords[c-1]
                    self.coords[c] = self.coords[c] + helix_transl
                continue
            else:
                first_H = True
            old_coord = self.coords[c].copy()
            self.coords[c] = d2_pos
            d2_pos = d1_pos
            d1_pos = old_coord
            prev_c = c
        return True

    def apply_corner_flip(self):
        # Find idx of corners
        diff1 = self.coords[1:] - self.coords[:-1]
        corner_bool = np.invert(np.all(np.equal(diff1[:-1], diff1[1:]), axis=1))
        # corner_bool = np.count_nonzero((self._coords[2:, :] - self._coords[:-2, :]), axis=1) == 1
        corner_bool[self.ss_sequence[1:-1] == 'H'] = False
        if not np.any(corner_bool): return False
        corner_idx = np.squeeze(np.argwhere(corner_bool), axis=1) + 1  # +1 as idx was of left neighbor
        np.random.shuffle(corner_idx)

        # Generate & check candidates
        for ci in corner_idx:
            candidate = self.corner_flip(self._coords[ci - 1, :3],
                             self._coords[ci, :3],
                             self._coords[ci + 1, :3])
            if not self.inNd(candidate)[0]:
            # if not nhp.inNd(candidate, self.coords)[0]:
                self._coords[ci, :] = candidate
                return True
        return False

    def apply_crankshaft_move(self):
        # temporarily shutdown: not sure how this contributes for BCC
        diff_4pos = self._coords[3:, :] - self._coords[:-3, :]  # Diff res with two spaces
        crank_bool = np.all(np.absolute(diff_4pos) == 2, axis=1)
        # crank_bool = np.sum(np.absolute(diff_4pos), axis=1) == 2  # if diff is 2 for that postion, it must be a u-loop
        if not np.any(crank_bool): return False
        crank_idx = np.squeeze(np.argwhere(crank_bool), axis=1)  # index of left-most position of the four points!
        np.random.shuffle(crank_idx)

        # Generate & check candidates
        for ci in crank_idx:
            crank_idx, crank_dir = abs(ci), copysign(1, ci)
            c0, c1, c2, c3 = self.coords[ci:ci + 4, :]
            c1_candidates = c0 + neighbor_mods
            c2_candidates = c3 + neighbor_mods
            c1_candidates = c1_candidates[np.invert(self.inNd(c1_candidates)), :]
            c2_candidates = c2_candidates[np.invert(self.inNd(c2_candidates)), :]
            if not len(c1_candidates) or not len(c2_candidates): continue
            np.random.shuffle(c1_candidates)
            for c1_candidate in c1_candidates:
                c2_idx = nhp.inNd(c2_candidates, c1_candidate + neighbor_mods)
                if np.any(c2_idx):
                    c2_candidates = c2_candidates[c2_idx]
                    np.random.shuffle(c2_candidates)
                    self._coords[ci + 1:ci + 3, :] = np.vstack((c1_candidate, c2_candidates[0]))
                    return True
        return False

    def set_hash_list(self):
        self.hash_list = set([hash(cc.tostring()) for cc in self.coords])

    def inNd(self, c):
        if c.ndim == 1:
            c = np.expand_dims(c, 0)
        c_hash_list = [hash(cc.tostring()) for cc in c]
        return [ch in self.hash_list for ch in c_hash_list]

    @staticmethod
    def branch_rotation(c, pivot, dim):
        """
        :param c: coordinates to change
        :param pivot: point around which to rotate
        :param dim: signed dimension in which to perform rotation (1, 2 or 3), pos for fwd, neg for rev
        :return: mutated coords
        """
        return np.dot(rotmats[dim], (c - pivot).T).T + pivot

    @staticmethod
    def corner_flip(c1, c2, c3):
        return c2 + ((c1 + c3) - 2 * c2)

    # --- stats and derived properties ----
    def get_pdb_coords(self, intermediate=False, conect_only=False):
        """
        Return coordinates in pdb format, as string
        :param intermediate: return without CONECT cards, required to create pdbs with multiple models
        :param conect_only: only return the CONECT cards
        :return:
        """
        coords_ca = self.coords - self.coords[0] # translate to 0,0,0
        coords_ca = coords_ca * self.lat_dist  # unit distances to real distances

        cn = (self.coords[1] - self.coords[0]) * -1 * sqrt(self.n1_dist ** 2 / 3)  # stick on N1 in opposite direction of chain
        cn_str = nhp.pdb_coord(cn)
        # resn = nhp.aa_dict[self.aa_sequence[0]]
        resn = nhp.aa_dict.get(self.aa_sequence[0], self.aa_sequence[0])
        txt = f'HETATM    1  N   {resn} A   1    {cn_str}  1.00  1.00           N\n'

        # Add CA coordinates
        an = 2  # atom number, start at 2 for first N
        an_alpha = 1  # tracker of alpha carbon atom number, just for CONECT record
        resi = 1

        conect = ""

        tag_coord_dict = {0: []}  # Fill in tag at pos 0, in case no other residues are tagged
        for ci in self.tagged_resi:
            if ci == 0: continue
            tag_coord_dict[ci], tag_coord_dict[0] = self.get_dye_coords(ci, 0)
        for ci, ca in enumerate(coords_ca):
            # --- add alpha carbon CA ---
            # resn_str = nhp.aa_dict[self.aa_sequence[ci]]
            resn_str = nhp.aa_dict.get(self.aa_sequence[ci], self.aa_sequence[ci])
            resi_str = str(resi).rjust(4)
            ca_str = nhp.pdb_coord(ca)
            txt += f'HETATM{str(an).rjust(5)}  CA  {resn_str} A{resi_str}    {ca_str}  1.00  1.00           C\n'
            conect += f"CONECT{str(an_alpha).rjust(5)}{str(an).rjust(5)}\n"
            an_alpha = an
            an += 1
            if ci in self.tagged_resi:  # Add tag residue
                if not len(tag_coord_dict[ci]): continue
                dye_coord = tag_coord_dict[ci]
                tc_str = nhp.pdb_coord(dye_coord[0])
                txt += f'HETATM{str(an).rjust(5)}  CB  {resn_str} A{resi_str}    {tc_str}  1.00  1.00           C\n'
                conect += f"CONECT{str(an_alpha).rjust(5)}{str(an).rjust(5)}\n"
                an += 1
            resi += 1

        # Add terminus
        an_str = str(an).rjust(5)
        resn_str = nhp.aa_dict.get(self.aa_sequence[-1], self.aa_sequence[-1])
        resi_str = str(resi - 1).rjust(4)  # - 1: still on same residue as last CA
        txt += f'TER   {an_str}      {resn_str} A{resi_str}\n'
        if conect_only:
            return conect
        elif intermediate:
            return txt
        return txt + conect

    def plot_structure(self, fn=None, auto_open=True):
        cols = [nhp.aa_to_hp_dict[aa] if resi not in self.tagged_resi else 'purple'
                for resi, aa in enumerate(self.aa_sequence)]
        trace_bb = go.Scatter3d(x=self.coords[:, 0],
                             y=self.coords[:, 1],
                             z=self.coords[:, 2],
                             line=dict(color=cols, width=20),
                             # marker=dict(size=5)
                                )
        trace_list = [trace_bb]
        pmin = np.min(self.coords)
        pmax = np.max(self.coords)
        layout = go.Layout(scene=dict(
            xaxis=dict(range=[pmin, pmax]),
            yaxis=dict(range=[pmin, pmax]),
            zaxis=dict(range=[pmin, pmax]),
            aspectmode='cube'
        )
        )
        fig = go.Figure(data=trace_list, layout=layout)
        if fn is None:
            py.offline.plot(fig, auto_open=auto_open)
        else:
            py.offline.plot(fig, filename=fn, auto_open=auto_open)

    def get_neighbors(self, c):
        return neighbor_mods + c

    # --- setters & properties ---
    def get_dye_coords(self, ci, partner_idx, expected_value=None):
        tag_obstructions_list = self.get_tag_obstructions(ci)
        unobstructed_tag_mods = [tm[0] / np.linalg.norm(tm[0]) for ti, tm in enumerate(tag_mods_single) if tag_obstructions_list[ti] == 0]
        if not len(unobstructed_tag_mods): return [], []

        ptc = (self.coords[partner_idx] - self.coords[ci])
        ptc = ptc / np.linalg.norm(ptc)
        tag_ca_dist = np.linalg.norm(self.coords[partner_idx] - self.coords[ci]) * self.lat_dist

        # Filter tag positions on angle
        angle_limit = 70 if tag_ca_dist <= 20 else 0
        angles = [nhp.get_angle(ptc, ut) for ut in unobstructed_tag_mods]
        ptc_angles_idx = [it for it, ut in enumerate(unobstructed_tag_mods) if angles[it] > angle_limit]
        if not len(ptc_angles_idx):
            ptc_angles_idx = [np.argmax(angles)]

        # Filter tag positions on dihedral
        # dist_best = np.Inf
        largest_dh = (-np.Inf, ())
        tuple_list = []
        tag0_obstructions_list = self.get_tag_obstructions(partner_idx)
        unobstructed_tag0_mods = [tm[0] / np.linalg.norm(tm[0]) for ti, tm in enumerate(tag_mods_single) if
                                  tag0_obstructions_list[ti] == 0]
        if not len(unobstructed_tag0_mods): return [], []
        for ti in ptc_angles_idx:
            for t0 in unobstructed_tag0_mods:
                dihedral = nhp.get_abs_dihedral(self.coords[ci], self.coords[0],
                                                self.coords[ci] + unobstructed_tag_mods[ti],
                                                self.coords[0] + t0)
                if dihedral > angle_limit:
                    tuple_list.append((unobstructed_tag_mods[ti], t0))
                if dihedral > largest_dh[0]:
                    largest_dh = (dihedral, (unobstructed_tag_mods[ti], t0))
                # dist = np.abs(dihedral - angles[ti])
                # if dist < dist_best:
                #     tuple_best = [unobstructed_tag_mods[ti], t0]
                #     dist_best = dist
        # if dist_best > 3: return [], []
        if len(tuple_list):
            tuple_best = random.choice(tuple_list)
        else:
            tuple_best = largest_dh[1]
        return [(self.coords[ci] - self.coords[0]) * self.lat_dist + tuple_best[0] * self.linker_dist], \
               [tuple_best[1] * self.linker_dist]

    @property
    def dist_fingerprint(self):
        if len(self.tagged_resi) < 2: return []
        fp = {}
        for fi in self.tagged_resi:
            if fi == 0: continue

            dye_coords, dye_coords_0 = self.get_dye_coords(fi, 0)
            if not len(dye_coords): continue
            cur_fp = []
            for d0 in dye_coords_0:
                cur_fp.extend([np.linalg.norm(d0 - dc) for dc in dye_coords])
            tt = self.tagged_resi_dict[fi]  # tag type
            if tt in fp:
                fp[tt].append(np.mean(cur_fp))
            else:
                fp[tt] = [np.mean(cur_fp)]
            # fp.append(np.mean(cur_fp))
        return fp

    @property
    def base_energy(self):
        return np.sum(self.individual_energies)

    @property
    def individual_energies(self):
        """
        Energy cost function
        """
        emat, e_wat, e_dsb, e_tag, e_reg = self.e_matrix
        e_aa = emat[:-4, :].sum().sum() / 2
        e_ss = emat[-4:-1, :].sum().sum()
        return e_aa, e_ss, e_wat, e_dsb, e_tag, e_reg

    def beta_sheet_bend_rule(self, c):
        # return np.sum(np.abs(c[2] - c[0]) == 4) > 1  # true if angles of 109.5 or 180 deg
        # return np.sum(np.abs(c[2] - c[0]) == 4) == 3  # true if angles 180 deg
        if len(c) == 2:
            return True
        return np.sum(np.abs(c[2] - c[0]) == 4) == 2  # true if angles 109.5 deg

    def beta_sheet_parallel_rule(self, neighbors, adjacents):
        parallel_dist = neighbors - adjacents
        inverse_dist = neighbors[::-1] - adjacents
        parallel_check = nhp.inNd(np.abs(parallel_dist[0]), quad_neighbor_mods_abs)[0] and len(np.unique(parallel_dist, axis=0)) == 1
        inverse_check = nhp.inNd(np.abs(inverse_dist[0]), quad_neighbor_mods_abs)[0] and len(np.unique(inverse_dist, axis=0)) == 1
        # parallel_check = np.all(nhp.inNd(np.abs(neighbors - adjacents), quad_neighbor_mods_abs))
        # inverse_check = np.all(nhp.inNd(np.abs(neighbors[::-1] - adjacents), quad_neighbor_mods_abs))
        if parallel_check or inverse_check:
            return True

    def get_tag_obstructions(self, ci):
        tag_obstructions_list = []
        for tm_bb, tm_bulk in tag_mods:
            bb_clashes = np.array(self.inNd(self.coords[ci] + tm_bb))
            bulk_coords = self.coords[ci] + tm_bulk
            bulk_clashes = np.array(self.inNd(bulk_coords))
            bulk_clashes[nhp.inNd(bulk_coords, self.coords[max(0, ci - 1):min(ci + 2, self.seq_length)])] = False
            tag_obstructions_list.append(np.sum(bb_clashes) + np.sum(bulk_clashes))
            # clash_bool = np.array(self.inNd(cur_coords))
            # clash_bool[nhp.inNd(cur_coords, self.coords[max(0, ci-1):min(ci+2, self.seq_length)])] = False
            # tag_obstructions_list.append(np.sum(clash_bool))

        # if ci == 0 or ci == self.seq_length - 1:
        #     tag_obstructions_list = [to - 2 for to in tag_obstructions_list]
        # else:
        #     tag_obstructions_list = [to - 3 for to in tag_obstructions_list]
        return tag_obstructions_list

    def get_contacts(self, ci):
        """
        Find contacting indices:
        - maximum 6
        - not including direct neighbors and self
        - First counting direct adjacents, then lv2 adjacents
        - not counting lv2 adjacents, obscured by lv1 adjacents
        """
        # Find which lv1 adajacents are matched
        contact_idx1 = np.argwhere(self.inNd(self.coords[ci] + neighbor_mods)).squeeze(-1)

        # remove lv2 adjacents behind occupied lv1 adjacent vertices
        idx_to_remove = [mod2mod_dict[cidx1] for cidx1 in contact_idx1]
        nm2_bool = np.ones(len(neighbor_mods2), dtype=bool)
        nm2_bool[idx_to_remove] = False
        contact_idx2 = np.argwhere(nhp.inNd(self.coords, self.coords[ci] + neighbor_mods2[nm2_bool])).squeeze(-1)

        # exclude indices of residue itself and up to 2 neighbors
        contact_idx2 = contact_idx2[np.logical_or(contact_idx2 > ci + 2, contact_idx2 < ci - 2)]

        # ensure no more than 6 contacts
        if len(contact_idx2) > 6:
            # Pick 6 closest (automatically ensures lv1 adjacents are picked)
            contact_idx2 = contact_idx2[np.argsort([np.linalg.norm(self.coords[ci] - self.coords[ci2]) for ci2 in contact_idx2])][:6]

        return contact_idx2

    @cached_property
    def e_matrix(self):
        """
        Energy cost function
        """
        # if self.experimental_mode == 13:  # only attempt to minimize difference expected and modeled tag distance
        #     tag0_coord = self.get_dye_coords(0)
        #     tag_coord = self.get_dye_coords(self.tagged_resi[1])
        #     if not len(tag0_coord) or not len(tag_coord):
        #         return np.zeros((self.seq_length+3, self.seq_length)), 0, 0, 1E10
        #     tag_dist = np.linalg.norm(tag0_coord[0] - tag_coord[0])
        #     e_tag = np.abs(exp_df.loc[self.pdb_id, 'distance'] - tag_dist) * 100
        #     return np.zeros((self.seq_length+3, self.seq_length)), 0, 0, e_tag

        seqi_list = np.arange(self.seq_length)
        e_wat = 0
        e_dsb = 0
        ss_multiplier = 25
        double_s_idx = []
        e_aa_array = np.zeros((len(seqi_list), len(seqi_list)), dtype=float)
        ss_array = np.tile('L', len(seqi_list))
        tag_array = np.zeros(len(seqi_list), dtype=float)
        outer_limits = np.vstack((self.coords.max(axis=0), self.coords.min(axis=0)))
        for ci, c in enumerate(self.coords):
            # If tagged, residue can't contribute to other terms and must be on outer border (heavy penalty)
            if ci in self.tagged_resi:
                tag_obstructions_list = self.get_tag_obstructions(ci)
                tag_array[ci] += 100 * min(tag_obstructions_list)
                continue

            # alpha helix h-bond contribution
            if ci < self.seq_length - 4:
                helix_candidate = self.coords[ci:ci + 5] - self.coords[ci]
                hel_dists = [np.linalg.norm(helix_candidate - hel) for hel in rotated_helix_array_list]
                if np.any(np.array(hel_dists) == 0):
                    ss_array[ci] = 'H'

            c_resn = self.aa_sequence[ci]
            contact_idx = self.get_contacts(ci)
            for cci in contact_idx:

                # res-res contribution
                e_aa_array[ci, cci] = self.pairs_mat.loc[c_resn, self.aa_sequence[cci]]
                if c_resn == 'C' and self.aa_sequence[cci] == 'C':
                    e_dsb -= 50  # disulfide bridge bonus

            # water contribution
            e_wat += (6 - len(contact_idx)) * self.pairs_mat.loc[c_resn, 'HOH']

            # beta-strand H-bonds
            if self.ss_sequence[ci] != 'S': continue  # Cannot form sheet with non-sheet residue
            nb_sheet_hbonds = 0
            confirmed_sheet_neighbor = None
            for idx in contact_idx:
                if self.ss_sequence[idx] != 'S': continue # Cannot form sheet with non-sheet residue
                if idx in self.sheet_block_dict and ci in self.sheet_block_dict:
                    if self.sheet_block_dict[idx] == self.sheet_block_dict[ci]: continue  # Cannot form sheet with residues from same contiguous strand
                if confirmed_sheet_neighbor is not None:
                    if confirmed_sheet_neighbor not in (ci-1, ci+1): continue  # cannot form sheet with neighbors of previously confirmed sheet bond
                ss_array[idx] = 'S'
                ss_array[ci] = 'S'
                confirmed_sheet_neighbor = idx
                nb_sheet_hbonds += 1
                if nb_sheet_hbonds == 2:
                    double_s_idx.append(ci)
                    break
        e_ss_array = np.zeros((3, len(seqi_list)), dtype=float)
        for ssi, ssc in enumerate(('H', 'S', 'L')):
            e_ss_array[ssi, ss_array == ssc] = self.ss_df.loc[ss_array == ssc, ssc] * ss_multiplier
        e_ss_array[1, double_s_idx] = e_ss_array[1, double_s_idx] * 2
        e_aa_array[np.tril_indices(len(seqi_list), -1)] = e_aa_array.T[np.tril_indices(len(seqi_list), -1)]

        if self.finetune_structure:
            imposer.set(self.coords, self.cm_coords)  # superimpose on center-of-mass coords
            imposer.run()
            e_tag = imposer.get_rms()
            e_ss_array[1, :] = 0  # set sheet modifier to 0
            e_out = np.row_stack((np.zeros((self.seq_length, self.seq_length)),
                                  e_ss_array,
                                  np.zeros(self.seq_length)
                                  ))
            return e_out, 0, 0, e_tag

        e_out = np.row_stack((e_aa_array, e_ss_array, tag_array))

        e_tag = tag_array.sum()
        if self.no_regularization:
            e_reg = 0
        else:
            # imposer.set(self.coords, self.prev_coords)  # superimpose on previous iteration coords
            # imposer.run()
            # e_reg = imposer.get_rms() * 25
            e_reg = np.sum(np.linalg.norm(self.coords - self.prev_coords, axis=1))
        return e_out, e_wat, e_dsb, e_tag, e_reg

    @property
    def rg(self):
        ca_coords = self.coords - self.coords[0]  # translate to 0,0,0
        ca_coords = ca_coords * sqrt((0.5 * self.ca_dist) ** 2 / 3)  # unit distances to real distances
        res_mass = np.expand_dims([nhp.aa_mass_dict.get(rn, 0) for rn in self.aa_sequence], -1)
        cm = ca_coords * res_mass
        tmass = np.sum(res_mass)
        rr = np.sum(cm * ca_coords)
        mm = np.sum(np.power(np.sum(cm, axis=0) / tmass, 2))
        rg2 = rr / tmass - mm
        if rg2 < 0:
            return 0.0
        return sqrt(rg2)

    # @property
    # def rg_old(self):
    #     coord_mods = ((self.coords[1:, :3] - self.coords[:-1, :3]) / 2).astype(int)
    #     cm_list = np.split(coord_mods, axis=0, indices_or_sections=coord_mods.shape[0])
    #     cm_coords = np.zeros_like(self.coords, dtype=float)
    #     cm_coords[0] = self.coords[0]
    #     for cmi, cm in enumerate(cm_list):
    #         aa1 = nhp.aa_dict.get(self.aa_sequence[cmi], self.aa_sequence[cmi])
    #         aa2 = nhp.aa_dict.get(self.aa_sequence[cmi + 1], self.aa_sequence[cmi + 1])
    #         if aa1 == 'TAG' or aa2 == 'TAG':
    #             d = self.ca_dist
    #         else:
    #             d = nhp.cm_dist_df.loc[aa1, aa2]
    #         cm_coords[cmi + 1, :] = cm_coords[cmi, :] + cm * sqrt((0.5 * d) ** 2 / 3)
    #     # cm_coords = self.coords[:, :3] * self.cm_dist  # note: commented, because unit coords are multiplied by modifiers above!
    #     res_mass = np.expand_dims([nhp.aa_mass_dict.get(rn, 0) for rn in self.aa_sequence], -1)
    #     cm = cm_coords * res_mass
    #     tmass = np.sum(res_mass)
    #     rr = np.sum(cm * cm_coords)
    #     mm = np.sum(np.power(np.sum(cm, axis=0) / tmass, 2))
    #     rg2 = rr / tmass - mm
    #     if rg2 < 0:
    #         return 0.0
    #     return sqrt(rg2)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, new_coords):
        if new_coords is None:
            if self.starting_structure == 'stretched':
                self._coords = self.stretched_init(self.seq_length)
        else:
            self._coords = np.copy(new_coords)
        self.set_hash_list()

    def stretched_init(self, seq_length):
        coords = np.zeros((seq_length, 3), dtype=int)
        for n in range(1,seq_length):
            mod = [-1, 1][n % 2]
            coords[n, :] = coords[n-1, :] + np.array([2 * mod, 2 * mod, 2])
        return coords
