import os, sys
from math import sqrt, copysign
import numpy as np
import helpers as nhp
from helpers import rotmat_dict
from LatticeModel import LatticeModel
from cached_property import cached_property
import random
import plotly as py
import plotly.graph_objs as go
from cached_property import cached_property



class Lattice(LatticeModel):
    """Class containing all that pertains to a particular type of lattice (initialization, allowed moves etc.)

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steric_hindrance_penalty = 55  # energy penalty for side chains at adjacent indices pointing the same way, Abeln 2014
        self.hb_penalty = -50  # energy bonus for forming an H bond, Abeln 2014
        self.ca_dist = 3.8  # Distance in A between CAs at adjacent coordinates
        self.linker_dist = 20  # Distance tagged CA to dye todo: wild guess based on linker, should MD!
        self.cacb_dist = 1.53  # Distance between CA and CB
        self.n1_dist = 1.48  # estimate of distance N to CA
        self.cm_dist = 4.75  # distance between adjacent centers of mass (deduced from human swissprot dataset)
        self.tag_dist = 25
        self.pairs_mat = kwargs['pairs_mat']
        self.pro_penalty = kwargs.get('pro_penalty', 0.0)
        self.sol_energy_factor = kwargs.get('sol_energy_factor', 1.0)
        self.p_global = kwargs.get('p_global', 0.1)

        self.no_anchoring = kwargs.get('no_anchoring', False)
        self.coords = kwargs.get('coords', None)
        self.res_coords_mod = kwargs.get('res_coords_mod', None)
        self.correct_side_chains()
        self.state = np.ones(self.seq_length, dtype=bool)  # start off with all in coil state == 1

    @property
    def rg(self):
        """
        Radius of gyration, based on the implementation for pymol: https://pymolwiki.org/index.php/Radius_of_gyration
        Uses mass of amino acid, centered at center of mass (actually ca-coords but spaced as if centers of mass)
        """
        coord_mods = self.coords[1:, :3] - self.coords[:-1, :3]
        cm_list = np.split(coord_mods, axis=0, indices_or_sections=coord_mods.shape[0])
        cm_coords = np.zeros_like(self.coords[:, :3], dtype=float)
        cm_coords[0, :] = self.coords[0, :3]
        for cmi, cm in enumerate(cm_list):
            aa1 = nhp.aa_dict[self.aa_sequence[cmi]]
            aa2 = nhp.aa_dict[self.aa_sequence[cmi + 1]]
            cm_coords[cmi + 1, :] = cm_coords[cmi, :] + cm * nhp.cm_dist_df.loc[aa1, aa2]
        # cm_coords = self.coords[:, :3] * self.cm_dist  # note: commented, because unit coords are multiplied by modifiers above!
        res_mass = np.expand_dims([nhp.aa_mass_dict[rn] for rn in self.aa_sequence], -1)
        cm = cm_coords * res_mass
        tmass = np.sum(res_mass)
        rr = np.sum(cm * cm_coords)
        mm = np.sum(np.power(np.sum(cm, axis=0) / tmass, 2))
        rg2 = rr / tmass - mm
        if rg2 < 0:
            return 0.0
        return sqrt(rg2)


    @cached_property
    def start_pos(self):
        """
        Placeholder vector for coordinates
        """
        # mid_point = self.lattice_dims[0] // 2
        # return np.tile((mid_point, mid_point, 0, 1), (self.seq_length, 1))
        return np.tile((0, 0, 0), (self.seq_length, 1))

    @property
    def individual_energies(self):
        """
        Energy cost function
        """
        e_aa = 0
        e_hb = 0
        has_neighbor_bool = nhp.inNd(self.res_coords, self.coords)

        # AA water contact contribution
        e_sol_vec = np.delete(np.invert(has_neighbor_bool) * self.pairs_mat.loc[self.aa_sequence, 'HOH'].to_numpy(),
                              self.tagged_resi)
        e_sol_vec[e_sol_vec < 0] *= self.sol_energy_factor
        e_sol = np.sum(e_sol_vec)
        # e_sol = np.sum(np.invert(has_neighbor_bool) * self.pairs_mat.loc[aa_seq_noTagged, 'HOH']) * self.sol_energy_factor

        # Steric hindrance contribution
        # sh_bool_fwd = np.all(self.res_coords_mod[1:, :] - self.res_coords_mod[:-1, :] == 0, axis=1)
        # sh_bool_rev = np.all(self.res_coords_mod[:-1, :] - self.res_coords_mod[1:, :] == 0, axis=1)
        # e_sh = np.sum(np.logical_or(sh_bool_fwd, sh_bool_rev)) * self.steric_hindrance_penalty
        sh_bool = np.all(self.res_coords_mod[1:, :] - self.res_coords_mod[:-1, :] == 0, axis=1)
        e_sh = np.sum(sh_bool) * self.steric_hindrance_penalty

        for ci in range(self.seq_length):

            cur_aa = self.aa_sequence[ci]
            neighbor_bool = np.sum(np.abs(self.coords[ci, :] - self.coords), axis=1) == 1
            if ci != 0: neighbor_bool[ci - 1] = False  # Direct sequential neighbors do NOT count
            if ci < self.seq_length - 1: neighbor_bool[ci + 1] = False

            # H-bond contribution
            if self.state[ci] == 0:
                resdir_signed_bool = nhp.inNd(self._res_coords_mod,
                                              self._res_coords_mod[ci, :])  # check: direction same
                hbond_disallowed_neighbor_coords = np.vstack((self.coords[ci, :] + self.res_coords_mod[ci, :],
                                                              self.coords[ci, :] - self.res_coords_mod[ci, :]))
                hbond_neighbor_bool = np.logical_and(neighbor_bool, np.invert(nhp.inNd(self.coords,
                                                                                       hbond_disallowed_neighbor_coords)))  # check: neighbor positions, but not in same dimension as side chain extends
                hbond_bool = np.logical_and(hbond_neighbor_bool, np.logical_and(resdir_signed_bool, np.invert(
                    self.state)))  # check: in beta-strand state
                e_hb += np.sum(hbond_bool) * self.hb_penalty * 0.5

            # AA contact contribution
            if ci in self.tagged_resi: continue  # tagged aa's can't form contacts
            if not np.any(neighbor_bool): continue
            ni = np.where(neighbor_bool)[0]  # neighbor index
            # case 1: CA--R  R--CA
            res_opposite_bool = nhp.inNd(self._res_coords_mod[ni, :], self._res_coords_mod[ci, :] * -1)
            res_on_bb_bool = nhp.inNd(self.coords[ni, :], self.res_coords[ci, :])
            e_bool1 = np.logical_and(res_on_bb_bool, res_opposite_bool)

            # case 2 parallel residues: CA^--R CA^--R
            res_parallel_bool = nhp.inNd(self._res_coords_mod[ni, :], self._res_coords_mod[ci, :])
            ca_dim = np.where(self.coords[ni, :3] != self.coords[ci, :3])[1]
            res_dim = np.array([np.argwhere(self.res_coords[nii, :3] != self.res_coords[ci, :3])[0, 0] for nii in ni])
            e_bool2 = np.logical_and(res_parallel_bool, ca_dim != res_dim)

            e_bool = np.logical_or(e_bool1, e_bool2)
            neighbor_aas = self.aa_sequence[ni[e_bool]]

            # res_neighbor_bool = np.sum(np.abs(self.res_coords[ci, :] - self.res_coords), axis=1) == 1
            # resmod_unsigned = np.row_stack((self._res_coords_mod[ci, :], self._res_coords_mod[ci, :] * -1))
            # resdir_unsigned_bool = nhp.inNd(self._res_coords_mod, resmod_unsigned)
            # e_bool = np.logical_and(neighbor_bool, np.logical_and(resdir_unsigned_bool, res_neighbor_bool))
            # e_bool = np.logical_and(~self.tagged_resi_bool, e_bool)
            # neighbor_aas = self.aa_sequence[e_bool]
            e_aa += sum([self.pairs_mat.loc[cur_aa, na] for na in neighbor_aas]) * 0.5

        return e_aa, e_hb, e_sol, e_sh

    @property
    def base_energy(self):
        return sum(self.individual_energies)

    @cached_property
    def tagged_resi_bool(self):
        bool_array = np.zeros(self.seq_length, dtype=bool)
        bool_array[self.tagged_resi] = True
        return bool_array



    @staticmethod
    def stretched_init(seq_length):
        """
        Generate a stretched configuration for a given number of residues
        """
        coords = np.zeros((seq_length, 3), dtype=int)
        coords[:, 2] += np.arange(seq_length)
        return coords

    # --- Global mutations ---

    @staticmethod
    def branch_rotation(c, pivot, dim):
        """
        :param c: coordinates to change
        :param pivot: point around which to rotate
        :param dim: signed dimension in which to perform rotation (1, 2 or 3), pos for fwd, neg for rev
        :return: mutated coords
        """
        return np.dot(rotmat_dict[dim], (c - pivot).T).T + pivot

    @staticmethod
    def corner_flip(c1, c2, c3):
        return c2 + ((c1 + c3) - 2 * c2)

    @staticmethod
    def crankshaft_move(c, direction):
        da = c[0, :] != c[3, :]  # dim in which hinge points differ
        db = np.all(c == c[0, :], axis=0)  # dim in which all points are same
        dc = np.logical_and(np.invert(da), np.invert(db))  # dim in which 'swing' differs from hinge points
        c[(1, 2), dc] = c[(0, 3), dc]
        c[(1, 2), db] = c[(1, 2), db] + direction
        return c[1:3, :]

    @staticmethod
    def get_neighbors(c, d=1):
        neighbors = np.tile(c, (6, 1))
        neighbors += np.row_stack((np.eye(3, dtype=int) * d, np.eye(3, dtype=int) * -1) * d)
        return neighbors

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, new_coords):
        """
        Set coords to newly provided coords if provided, or set with a random walk
        """
        # self._coords = self.start_pos
        if new_coords is None:
            nb_attempts = 500
            # Allowed types:
            # - stretched
            # - serrated
            # - free_random
            # - anchored_random
            if self.starting_structure == 'stretched':
                self._coords = self.stretched_init(self.seq_length)
                return
            # if self.starting_structure == 'free_random':
            #     for attempt_idx in range(nb_attempts):
            #         if self.perform_random_walk([]): return
            # elif self.starting_structure == 'anchored_random':
            #     anchors = [idx for idx, aa in enumerate(self.aa_sequence) if idx in self.tagged_resi]
            #     for attempt_idx in range(nb_attempts):
            #         if self.perform_random_walk(anchors): return
            raise ValueError(f'No feasible random start state reached in {nb_attempts} attempts!')
        else:
            self._coords = new_coords

    @property
    def res_coords(self):
        """
        Residue coords
        """
        return self.coords + self._res_coords_mod

    @property
    def res_coords_mod(self):
        """
        Modifier for residue coords, add to CA coords to get residue coords
        """
        return self._res_coords_mod

    @property
    def res_coords_plottable(self):
        """
        residue coords, fit for plotting only; shorter distances for untagged, arbitrarily longer for tagged
        """
        coords_mod = self._res_coords_mod * 0.3
        coords_mod[self.tagged_resi, :] *= 20
        return self.coords + coords_mod

    def get_distmat(self, coords, anchors):
        """
        return the distances between all lattice points and a number of anchor coordinates
        """
        if anchors.ndim == 1:
            anchors = np.expand_dims(anchors, 0)
        return np.column_stack([np.sum(np.abs(coords - an), axis=1) for an in anchors])

    def get_free_neighbor_coords(self, c):
        """
        Return free lattice vertices adjacent to given coordinates c, or empty vector if none exist
        """
        neighbors = nhp.get_neighbors(c)
        neighbor_bool = np.invert(nhp.inNd(neighbors[:, :3], self.coords[:, :3]))
        return neighbors[neighbor_bool, :]

    def get_adjacent_bb_coords(self, idx):
        """
        Get coordinates of CAs at positions adjacent to CA at given index idx
        """
        if idx == 0:
            return self.coords[idx:idx + 2, :]
        elif idx == self.seq_length - 1:
            return self.coords[idx - 1:idx, :]
        else:
            return self.coords[idx - 1:idx + 2, :]

    @res_coords_mod.setter
    def res_coords_mod(self, new_coords):
        self._res_coords_mod = np.zeros((self.seq_length, 3), dtype=int)
        if new_coords is None:
            mod_idx = np.random.randint(3, size=self.seq_length)
            for i in enumerate(mod_idx): self.res_coords_mod[i] = 1
        else:
            self._res_coords_mod[:] = new_coords

    # --- Methods to set initial position residue coordinates ---
    def perform_random_walk(self, anchors):
        """
        Set residue coordinates with a random walk
        :return: True if random walk was succesful, False if walk ended prematurely n times.
        """
        nb_attempts = 500
        prev_anchor = 0
        if 0 in anchors: anchors.remove(
            0)  # No need to set/anchor coords first residue; by default set at one lattice edge
        first = True
        anchoring = False
        if anchoring:
            for an in anchors:
                for attempt_idx_1 in range(nb_attempts):
                    anchors_dist = an - prev_anchor
                    # Anchor a tagged position to a lattice edge
                    route_success = self.set_edge_coord(an, anchor=self.coords[prev_anchor, :],
                                                        dist=anchors_dist, first=first)
                    # Find a route between anchors
                    if route_success:
                        route_success = self.set_new_coords(prev_anchor + 1, an - 1, anchored=True)
                    # If both succeed, continue to next anchor
                    if route_success:
                        first = False
                        break
                    if attempt_idx_1 == nb_attempts - 1:
                        return False
                prev_anchor = an
        for attempt_idx_2 in range(nb_attempts):  # set route of protein after last anchor
            route_success = self.set_new_coords(prev_anchor + 1, self.seq_length - 1, anchored=False)
            if route_success: break
            if attempt_idx_2 == nb_attempts - 1:
                return False

        # todo: test, grow lattice to allow free movement after initial constraint
        self._lattice_dims = [self.seq_length, self.seq_length, self.seq_length]

        return True

    def get_edge_candidates(self, limits_bool, anchor, dist):
        """
        Find an extreme coordinate (i.e. highest/lowest value in any direction) to initially put a tagged residue at.

        :param limits_bool: 2 x 3 matrix, indicating in which dims (columns) and direction (rows) previous anchor is at
            edge of current structure (i.e. where new anchor shouldn't be at edge).
        :param anchor: previous anchor cooordinates
        :param dist: nb of residues to leave between returned position and previous anchor
        :return: candidate coordinates at distance dist from anchor, at edge of structure
        """
        # convert limits_bool to index array stating [[direction, dim],...] in which to grow
        limits_idx = np.vstack(np.where(np.invert(limits_bool))).T
        # lat_limits = np.array(([0, 0, 0], self.lattice_dims))
        #
        # # Option 1: Get a free lattice edge vertex within range
        # anchor_bool = np.sum(np.abs(self.free_coords - anchor[:3]), axis=1) <= dist  # Check: within dist of anchor
        # # Check: on lattice edges determined by limits_idx
        # prev_bool = np.zeros(self.free_coords.shape[0], dtype=bool)
        # for dir, dim in limits_idx:
        #     temp_bool = self.free_coords[:, dim] >= lat_limits[dir, dim] if dir else \
        #         self.free_coords[:, dim] <= lat_limits[dir, dim]
        #     prev_bool = np.logical_or(temp_bool, prev_bool)
        #
        # return_bool = np.logical_and(prev_bool, anchor_bool)
        # if np.any(return_bool):
        #     return self.free_coords[return_bool, :3]

        # Option 2: get free extreme coordinate not on lattice edge but within range

        # determine new direction to grow in
        nli = random.randint(0, limits_idx.shape[0] - 1)
        dir, dim = limits_idx[nli, :]

        # Generate all valid candidate points
        mod = np.mgrid[-dist:dist, -dist:dist, -dist:dist].T.reshape(-1, 3)
        mod = mod[np.sum(np.abs(mod), axis=1) <= dist, :]  # at correct distance
        candidates = anchor[:3] + mod
        candidates = candidates[(candidates[:, dim] > anchor[dim])] if dir \
            else candidates[(candidates[:, dim] < anchor[dim])]  # in correct dimension

        # non-negative # todo kept to simulate a protein being fixed on a slide, but that could be done better I guess
        candidates = candidates[np.invert(np.any(candidates < 0, axis=1)), :]
        # # Check if within limits  # todo not sure why this was still there...
        # inlattice_bool = nhp.inNd(candidates, self.free_coords)
        # return candidates[inlattice_bool, :]
        return candidates

    def set_edge_coord(self, idx, anchor, dist, first=False):
        """
        set coordinates for res at idx at a random position at the edge of the lattice.
        Provide the previous anchor point and the distance between this anchor and this
        point as restraint. Next edge coord will be set a random limit.
        """
        # Find which limits previous anchor was at
        if first:
            limits_bool = np.array([[False, False, True],
                                    [False, False, False]], dtype=bool)
        else:
            limits = np.percentile(self.coords[:, :3], [0, 100], axis=0)
            limits_bool = np.vstack([np.equal(lim.squeeze(), anchor[:3]) for lim in np.split(limits, 2, axis=0)])
        # Find possible modifiers in allowed directions
        candidates = self.get_edge_candidates(limits_bool, anchor, dist)
        # If no options remain (e.g. tags too close to eachother), try the other dim/dir combinations
        if candidates.size == 0:
            candidates = self.get_edge_candidates(np.invert(limits_bool), anchor, dist)
        if candidates.size == 0:
            return False
        coords = nhp.pick_random_coords(candidates)
        self._coords[idx, :] = np.concatenate((coords, [1]))
        return True

    def set_new_coords(self, idx_start, idx_end, anchored=False):
        """
        Set valid random positions for a range of indices. Optionally, anchor between the adjacent positions.
        :return: bool, True if route was set succesfully, False if route ran out of free paths before finishing.
        """
        route_success = True
        if anchored:
            try:
                next_anchor_idx = idx_end + 1
            except:
                pass
            previous_anchor_idx = idx_start - 1
            anchors_dist = next_anchor_idx - previous_anchor_idx - 1
        if idx_end < idx_start:  # happens when two anchors are adjacent
            return True

        idx_list = list(range(idx_start, idx_end + 1))
        # Set coordinates
        for iidx, idx in enumerate(idx_list):
            candidates = self.get_free_neighbor_coords(self._coords[idx - 1, :])  # ...free (and within lattice)
            if candidates.size == 0:
                route_success = False
                break
            if anchored:
                dm = self.get_distmat(candidates, anchors=self._coords[next_anchor_idx, :])
                candidates = candidates[dm[:, 0] <= anchors_dist - iidx, :]  # ...with enough steps left to reach end
                if candidates.size == 0:
                    route_success = False
                    break
            self._coords[idx, :] = nhp.pick_random_coords(candidates)

        # If ran stuck: reset coordinates to zeros
        if not route_success:
            self._coords[idx_list, :] = np.zeros_like(self._coords[idx_list, :])
            return False
        return True

    # --- Mutations ---
    def is_valid_candidate_single(self, candidate):
        """
        Check which of given candidates is valid. Last two dims are always considered the number of residues and
        coordinates per residue respectively.
        :candidates: m x n ... x seq_length x 4 array
        :change_idx: list of len m x n x ... with indices of changed residues for this candidate
        :return: m x n ... np boolean array, True if candidate is valid
        """
        # idx_array = np.meshgrid(*[list(range(0, n)) for n in candidates.shape[:-2]], indexing='ij')
        # idx_array = np.column_stack([idx.reshape(-1, 1) for idx in idx_array])
        # bool_array = np.zeros(candidates.shape[:-2], dtype=bool) # Default validity as 'False'
        #
        # for idx in idx_array:
        #   candidate = candidates[tuple(idx)]
        turn_bool = np.any(np.abs(candidate[:-2, :] - candidate[2:, :]) == 2, axis=1)
        turn_bool = np.concatenate(([True], turn_bool, [True]))
        if np.any(np.logical_and(np.invert(self.state), turn_bool)):
            return False  # check: no bends at beta coords, else keep 'False'
        elif nhp.contains_double_coords(candidate):
            return False  # check: no double coords, else keep 'False'
        elif not self.is_unobstructed(candidate):
            return False
        return True  # If all checks passed: valid candidate

    def is_valid_candidate(self, candidates):
        """
        Check which of given candidates is valid. Last two dims are always considered the number of residues and
        coordinates per residue respectively.
        :candidates: m x n ... x seq_length x 4 array
        :change_idx: list of len m x n x ... with indices of changed residues for this candidate
        :return: m x n ... np boolean array, True if candidate is valid
        """
        idx_array = np.meshgrid(*[list(range(0, n)) for n in candidates.shape[:-2]], indexing='ij')
        idx_array = np.column_stack([idx.reshape(-1, 1) for idx in idx_array])
        bool_array = np.zeros(candidates.shape[:-2], dtype=bool)  # Default validity as 'False'

        for idx in idx_array:
            candidate = candidates[tuple(idx)]
            turn_bool = np.any(np.abs(candidate[:-2, :] - candidate[2:, :]) == 2, axis=1)
            turn_bool = np.concatenate(([True], turn_bool, [True]))
            if np.any(np.logical_and(np.invert(self.state),
                                     turn_bool)): continue  # check: no bends at beta coords, else keep 'False'
            if nhp.contains_double_coords(candidate): continue  # check: no double coords, else keep 'False'
            if not self.is_unobstructed(candidate): continue
            # if not self.is_within_lattice(candidate): continue
            bool_array[tuple(idx)] = True  # If all checks passed this loop: valid candidate
        return bool_array

    def is_unobstructed(self, coords):
        """
        Check whether tagged residues can reach edge of lattice unobstructed in a given FULL set of coordinates.
        """
        if self.no_anchoring:
            return True
        coord_ranges = np.row_stack((np.min(coords, axis=0), np.max(coords, axis=0)))
        for ti in self.tagged_resi:
            ray_bool = False
            ray_list = self.to_end(coords[ti, :], coord_ranges)
            for ray in ray_list:
                if not np.any(nhp.inNd(ray, coord_ranges)):
                    ray_bool = True
                    break
            if not ray_bool: return False
        return True

    def to_end(self, coord, ranges):
        """
        get coordinates of all lattice vertices from a point until an edge of the lattice
        :param d:
        :return:
        """
        out_list = []
        for d in range(3):
            ray_fw = np.array([]) if coord[d] == ranges[1, d] else np.arange(coord[d] + 1, ranges[1, d] + 1)
            coord_fw = np.tile(coord, (ray_fw.size, 1))
            coord_fw[:, d] = ray_fw
            out_list.append(coord_fw)
            ray_rev = np.array([]) if coord[d] == ranges[0, d] else np.arange(ranges[0, d], coord[d])
            coord_rev = np.tile(coord, (ray_rev.size, 1))
            coord_rev[:, d] = ray_rev
            out_list.append(coord_rev)
        return out_list

    # --- local mutations ---
    def apply_side_chain_move(self, mut_idx=None):
        """
        Find all valid positions for side chain moves and apply one at random
        :return: bool, True if mutation was succesful, False otherwise
        """
        # Pick a random AA
        if mut_idx is None:
            candidate_idx = np.arange(self.seq_length)[self.state]
            if not len(candidate_idx): return False
            mut_idx = random.choice(candidate_idx)
        if mut_idx in self.tagged_resi and not self.no_anchoring:  # Treat tagged residues differently; must always point outward
            coord_ranges = np.row_stack((np.min(self.coords, axis=0), np.max(self.coords, axis=0)))
            ray_list = self.to_end(self.coords[mut_idx], coord_ranges)
            ray_bool = np.ones(len(ray_list), dtype=bool)
            for ri, ray in enumerate(ray_list):
                if any(nhp.inNd(ray, self.coords)): ray_bool[ri] = False
            ray_idx = np.squeeze(np.argwhere(ray_bool), -1)
            if ray_idx.size == 0:
                return False
            cidx = np.random.choice(ray_idx)
            new_res_mod = np.zeros(3, dtype=int)
            new_res_mod[cidx // 2] = (-1) ** (cidx % 2)
            self._res_coords_mod[mut_idx, :] = new_res_mod
            return True
        candidate_res_coords = self.get_neighbors(self.coords[mut_idx, :])
        np.random.shuffle(candidate_res_coords)
        bb_coords = np.vstack((self.get_adjacent_bb_coords(mut_idx), self.res_coords[mut_idx, :]))
        for candidate in candidate_res_coords:
            if nhp.inNd(candidate, bb_coords)[0]: continue
            self._res_coords_mod[mut_idx, :] = candidate - self.coords[mut_idx, :]
            return True
        return False
        # res_coord = self.res_coords[mut_idx, :]
        # forbidden_coords = np.row_stack((res_coord, bb_coords))
        # allowed_bool = np.invert(nhp.inNd(candidate_res_coords, forbidden_coords))
        # crc = candidate_res_coords[allowed_bool, :]  # candidate residue coords
        # new_res_coord = crc[np.random.randint(crc.shape[0]), :]
        # self._res_coords_mod[mut_idx, :] = new_res_coord - self.coords[mut_idx, :]
        # return True

    def correct_side_chains(self, idx_list=None):
        """
        Check if side chain positions overlap with backbone, if so, correct
        :param idx_list: optionally, provide list of indices of residues to check. Default: checks all
        """
        if idx_list is None:
            idx_list = range(self.seq_length)
        for idx in idx_list:
            if nhp.inNd(self.res_coords[idx, :], self.get_adjacent_bb_coords(idx))[0]:
                self.apply_side_chain_move(idx)  # For pivot position

    def apply_state_change(self):
        """
        Find all allowed state changes, apply one
        """
        # prerequisites for flip to strand state:
        # no turn
        # if neighbor in strand state, residue must point in opposite direction
        turn_bool = np.any(np.abs(self.coords[:-2, :] - self.coords[2:, :]) == 2, axis=1)
        turn_bool = np.concatenate(([True], turn_bool, [True]))
        left_bool = np.logical_or(self.state[:-1],
                                  np.all(
                                      np.equal(self._res_coords_mod[:-1, :3], np.invert(self._res_coords_mod[1:, :3])),
                                      axis=1))
        left_bool = np.concatenate(([True], left_bool))
        right_bool = np.logical_or(self.state[1:],
                                   np.all(
                                       np.equal(self._res_coords_mod[1:, :3], np.invert(self._res_coords_mod[:-1, :3])),
                                       axis=1))
        right_bool = np.concatenate((right_bool, [True]))
        valid_beta_idx = np.argwhere(np.logical_and(turn_bool, np.logical_and(left_bool, right_bool)))
        # valid_beta_idx = np.column_stack((valid_beta_idx, np.repeat('b', valid_beta_idx.size)))

        # prereq to flip to alpha: must be beta
        valid_alpha_idx = np.argwhere(np.invert(self.state))
        if valid_alpha_idx.size == 0:
            return False
        mut_idx = random.choice(np.concatenate((valid_alpha_idx, valid_beta_idx)))
        self.state[mut_idx] = np.invert(self.state[mut_idx])
        return True

    # --- global mutations ---
    def apply_branch_rotation(self):
        """
        Find valid positions to apply branch rotation, apply one at random. Changes anywhere between all-1 to 1
        position(s).
        :return: bool, True if mutation was successful, False otherwise
        """
        mutations = list(range(-3, 4))
        mutations.remove(0)
        random.shuffle(mutations)  # randomize possible mutations once for entire attempt
        candidate_found = False
        idx_list = list(range(self.seq_length - 1))
        random.shuffle(idx_list)  # randomize positions to check
        for ci in idx_list:  # omit last position, where rotation does not make sense
            candidate = np.copy(
                self._coords)  # requires new copy each idx, as some non rotated postions might have changed
            for mi, mut in enumerate(mutations):
                candidate[ci + 1:, :] = self.branch_rotation(self._coords[ci + 1:, :], self._coords[ci, :], mut)
                if self.is_valid_candidate_single(candidate):
                    candidate_found = True
                    break
            if candidate_found:
                break
        if not candidate_found: return False

        # Adapt backbone and residual coords
        self.coords = candidate
        self._res_coords_mod[ci + 1:, :] = self.branch_rotation(self._res_coords_mod[ci + 1:, :], self._coords[ci, :],
                                                               mut)
        # self.correct_side_chains([ci])
        return True

    # @nb.njit(parallel=True)
    def apply_corner_flip(self):
        """
        Find valid positions to apply a corner flip and apply one at random. Changes 1 position.
        :return: bool, True if mutation was successful, False otherwise
        """
        # Find idx of corners: 2 points spaced by 1 point that differ in 2 coordinates
        corner_bool = np.count_nonzero((self._coords[2:, :] - self._coords[:-2, :]), axis=1) == 2
        if not np.any(corner_bool): return False
        corner_idx = np.squeeze(np.argwhere(corner_bool), axis=1) + 1  # +1 as idx was of left neighbor
        np.random.shuffle(corner_idx)
        # Generate candidates
        # candidates = np.tile(self._coords, (corner_idx.size, 1, 1))  # corner_idx x coords x dims
        candidate = np.copy(self._coords)
        candidate_found = False
        for ci in corner_idx:
            old_coords = np.copy(candidate[ci, :3])
            candidate[ci, :3] = self.corner_flip(self._coords[ci - 1, :3],
                                                self._coords[ci, :3],
                                                self._coords[ci + 1, :3])
            if self.is_valid_candidate_single(candidate):
                candidate_found = True
                break
            else:
                candidate[ci, :3] = old_coords
        if not candidate_found: return False
        self.coords = candidate
        # self.correct_side_chains(range(ci - 1, ci + 2))
        return True

    def apply_crankshaft_move(self):
        """
        Find valid positions to apply a crank shaft move and apply one at random. Changes 2 positions.
        :return: bool, True if mutation was successful, False otherwise
        """
        # Find idx where crank shaft is allowed; a u-shaped loop of 4 AAs.
        diff_4pos = self._coords[3:, :] - self._coords[:-3, :]
        crank_bool = np.sum(np.absolute(diff_4pos), axis=1) == 1  # if diff is 1 for that postion, it must be a u-loop
        if not np.any(crank_bool): return False
        crank_idx = np.squeeze(np.argwhere(crank_bool), axis=1)  # index of left-most position of the four points!
        crank_idx = np.concatenate((crank_idx, crank_idx * -1))  # may swing either way, so double the index
        np.random.shuffle(crank_idx)

        # generate candidates
        candidate = np.copy(self.coords)
        candidate_found = False
        for ci in crank_idx:
            ci_abs = abs(ci)
            old_coords = np.copy(self.coords[ci_abs + 1:ci_abs + 3, :])
            candidate[ci_abs + 1:ci_abs + 3, :] = self.crankshaft_move(np.copy(self.coords[ci_abs:ci_abs + 4, :]),
                                                                      copysign(1, ci))
            if self.is_valid_candidate_single(candidate):
                candidate_found = True
                break
            else:
                candidate[ci_abs + 1:ci_abs + 3, :] = old_coords
        if not candidate_found: return False

        self.coords = candidate
        # self.correct_side_chains(range(ci_abs, ci_abs + 4))
        return True

    def apply_n_steps(self, n):
        """
        Apply n mutations to the structure, chosen at random.
        :param n: number of mutations to use
        :return: boolean, False if no mutations can be made anymore, else True
        """
        global_fun_list = [self.apply_crankshaft_move,
                           self.apply_branch_rotation,
                           self.apply_corner_flip]
        local_fun_list = [self.apply_side_chain_move, self.apply_state_change]
        for _ in range(n):
            random.shuffle(global_fun_list)
            random.shuffle(local_fun_list)

            if not local_fun_list[0]():
                _ = local_fun_list[1]()  # local mutations cannot fail for now

            if random.uniform(0, 1) < self.p_global:
                if global_fun_list[0]():
                    self.correct_side_chains(); continue
                elif global_fun_list[1]():
                    self.correct_side_chains(); continue
                elif global_fun_list[2]():
                    self.correct_side_chains(); continue
                else:
                    return False

        return True

    def get_N1(self, real_ca1):
        """
        get position of first N atom in a free lattice direction
        """
        neighbor_coords = nhp.get_neighbors(self.coords[0, :])
        neighbor_coords_bool = nhp.inNd(neighbor_coords, self.get_adjacent_bb_coords(0))
        neighbor_coords_free = neighbor_coords[np.invert(neighbor_coords_bool), :]

        # prefer the direction opposite bond with C
        nc_test = self._coords[0, :] + (self._coords[1, :] - self._coords[0, :]) * -1
        if nhp.inNd(np.expand_dims(nc_test, 0), neighbor_coords_free)[0]:
            nc_out = real_ca1 + ((self._coords[1, :] - self._coords[0, :]) * -1 * self.n1_dist)[:3]
            return nc_out
        raise ValueError('position of N-terminus is not free!')
        # # If not free, pick a random one todo: shouldn't happen
        # ncf_choice = np.squeeze(random.choice(np.split(neighbor_coords_free, neighbor_coords_free.shape[0], axis=0)))
        # nc_out = np.copy(real_ca1) + ((ncf_choice - self._coords[0, :]) * self.n1_dist)[:3]
        # return nc_out

    @property
    def dist_fingerprint(self):
        """
        Return distances in A of tagged residues to N-terminus as list of floats
        """
        fingerprint = []
        tag_coords = self.coords[self.tagged_resi, :3] * self.ca_dist + self.res_coords_mod[self.tagged_resi,
                                                                        :3] * self.linker_dist
        for i in range(1, len(self.tagged_resi)):
            fingerprint.append(float(np.linalg.norm(tag_coords[0, :] - tag_coords[i, :])))
        return fingerprint

    def get_pdb_coords(self):
        """
        Return coordinates in model as PDB-format string (no REMARKS or END)
        """
        # translate and rotate
        coords_ca = self.coords[:, :3] - self.coords[0, :3]
        dir = np.argwhere(coords_ca[1]).squeeze()[()]
        if dir != 2:
            dir_rot = 2 - dir
            coords_ca = np.matmul(coords_ca, nhp.get_rotmat(dir_rot)[:3,:3])
        coords_ca = coords_ca * self.ca_dist  # unit distances to real distances
        # coords_ca = (self.coords[:, :3] + self.coords[:, :3].min().__abs__()) * self.ca_dist
        coords_cb = coords_ca + self.res_coords_mod[:, :3] * self.cacb_dist

        cn = self.get_N1(coords_ca[0, :])
        cn_str = nhp.pdb_coord(cn)
        resn = nhp.aa_dict[self.aa_sequence[0]]
        txt = f'HETATM    1  N   {resn} A   1    {cn_str}  1.00  1.00           N\n'

        # Add CA of other residues
        ci = 0
        # cip = 1
        an = 2  # atom number, start at 2 for first N
        an_alpha = 1  # tracker of alpha carbon atom number, just for CONECT record
        resi = 1

        conect = ""
        for ci, (ca, cb) in enumerate(zip(coords_ca, coords_cb)):
            # --- add alpha carbon CA ---
            resn_str = nhp.aa_dict[self.aa_sequence[ci]]
            resi_str = str(resi).rjust(4)
            ca_str = nhp.pdb_coord(ca)
            txt += f'HETATM{str(an).rjust(5)}  CA  {resn_str} A{resi_str}    {ca_str}  1.00  1.00           C\n'
            conect += f"CONECT{str(an_alpha).rjust(5)}{str(an).rjust(5)}\n"
            an_alpha = an
            an += 1
            if resn_str != 'GLY':
                # --- add beta carbon CB ---
                cb_str = nhp.pdb_coord(cb)
                txt += f'HETATM{str(an).rjust(5)}  CB  {resn_str} A{resi_str}    {cb_str}  1.00  1.00           C\n'
                conect += f"CONECT{str(an_alpha).rjust(5)}{str(an).rjust(5)}\n"
                an += 1
            if ci in self.tagged_resi:
                # --- add placeholder atom for tag CT ---
                ct_str = nhp.pdb_coord(ca + self.res_coords_mod[ci, :3] * self.tag_dist)
                txt += f'HETATM{str(an).rjust(5)}  CT  {resn_str} A{resi_str}    {ct_str}  1.00  1.00           C\n'
                conect += f"\nCONECT{str(an-1).rjust(5)}{str(an).rjust(5)}\n"
                an += 1
            resi += 1

        # Add terminus
        an_str = str(an).rjust(5)
        resn_str = nhp.aa_dict[self.aa_sequence[-1]]
        resi_str = str(resi - 1).rjust(4)  # - 1: still on same residue as last CA
        txt += f'TER   {an_str}      {resn_str} A{resi_str}\n'
        txt += conect
        return txt

    def save_pdb(self, fn, rg_list=None):
        """
        write current latice model to a readable pdb format file
        :param fn: file name
        :param rg_list: list of radius of gyration values, to store in pdb remarks 0
        """
        txt = f'REMARK   0 BASE ENERGY {self.base_energy}\n'
        if rg_list is not None:
            for rg in rg_list:
                txt += f'REMARK   1 RG {rg}\n'
        txt += self.get_pdb_coords()
        txt += '\n\nEND'

        with open(fn, 'w') as fh:
            fh.write(txt)

    def plot_structure(self, fn=None, auto_open=False):
        cols = [nhp.aa_to_hp_dict[aa] if resi not in self.tagged_resi else 'purple' for resi, aa in enumerate(self.aa_sequence)]
        trace_bb = go.Scatter3d(x=self.coords[:, 0],
                             y=self.coords[:, 1],
                             z=self.coords[:, 2],
                             line=dict(color=cols, width=20),
                             # marker=dict(size=5)
                                )

        trace_list = [trace_bb]
        for n in range(self.seq_length):
            cur_trace = go.Scatter3d(
                x=(self.coords[n, 0], self.res_coords_plottable[n, 0]),
                y=(self.coords[n, 1], self.res_coords_plottable[n, 1]),
                z=(self.coords[n, 2], self.res_coords_plottable[n, 2]),
                line=dict(color=cols[n], width=20),
                marker=dict(size=0))
            trace_list.append(cur_trace)
        all_points = np.row_stack((self.coords, self.res_coords_plottable))[:, :3]
        pmin = np.min(all_points)
        pmax = np.max(all_points)
        layout = go.Layout(scene=dict(
            xaxis=dict(range=[pmin, pmax]),
            yaxis=dict(range=[pmin, pmax]),
            zaxis=dict(range=[pmin, pmax])
        )
        )
        fig = go.Figure(data=trace_list, layout=layout)

        if fn is None:
            py.offline.plot(fig, auto_open=auto_open)
        else:
            py.offline.plot(fig, filename=fn, auto_open=auto_open)
