from random import uniform
import numpy as np
from copy import deepcopy
import importlib
import multiprocessing as mp
from time import sleep

from LatticeModel import LatticeModel

class LatticeModelComparison(object):
    def __init__(self, **kwargs):
        self.lattice_class = kwargs['lattice_type']
        self.beta = kwargs['beta']
        self.nb_steps = kwargs['nb_steps']
        self.mod_id = kwargs['mod_id']
        self.store_rg = kwargs['store_rg']
        # self.iter = 0
        self.kwargs = kwargs
        lm = kwargs.get('lm', None)
        if lm is not None:
            self.best_model, self.candidate_model = deepcopy(lm), deepcopy(lm)
        else:
            self.initialize()
        self.stepwise_intermediates = kwargs.get('stepwise_intermediates', None)

    @property
    def stepwise_intermediates(self):
        return self._stepwise_intermediates

    @stepwise_intermediates.setter
    def stepwise_intermediates(self, fn):
        if fn is None:
            self._stepwise_intermediates = fn
            return
        self._stepwise_intermediates = fn

        # Also allow for saving emats
        self.e_mat_dict = {'sequence': self.best_model.aa_sequence,
                           '0': self.best_model.e_matrix[0]}

    @property
    def delta_e(self):
        return self.best_model.base_energy - self.candidate_model.base_energy

    @property
    def lattice_class(self):
        return self._lattice_class

    @property
    def p_accept(self):
        if np.isnan(self.beta):
            if self.delta_e < 0:
                return 0 # candidate worse --> always reject
            else:
                return 1 # candidate better --> always accept
        return min((1, np.exp(min(709.7, self.delta_e * self.beta))))

    @property
    def store_rg(self):
        return self._store_rg

    @store_rg.setter
    def store_rg(self, srg):
        self._store_rg = False
        if srg == 'full':
            self.rg_list = []
            self._store_rg = True

    @lattice_class.setter
    def lattice_class(self, lattice_name):
        self._lattice_class = importlib.import_module(f'lattice_types.{lattice_name}').Lattice

    def accept_or_reject(self, replace=True, free_sampling=False):
        if free_sampling or uniform(0,1) < self.p_accept: # accept: candidate becomes best
            self.best_model.prev_coords[:] = self.best_model.coords  # required for regularization
            self.candidate_model.prev_coords[:] = self.best_model.coords
            self.best_model._coords[:] = self.candidate_model._coords
            self.best_model.set_hash_list()
            self.best_model.__dict__.pop('e_matrix', None)
            # self.best_model._res_coords_mod[:] = self.candidate_model._res_coords_mod
            return True
        else:  # reject: reset candidate
            if replace:
                self.candidate_model._coords[:] = self.best_model._coords
                self.candidate_model.set_hash_list()
                self.candidate_model.__dict__.pop('e_matrix', None)
                # self.candidate_model._res_coords_mod[:] = self.best_model._res_coords_mod
            return False

    def do_mc(self, nb_iterations, silent=False):
        """
        Perform a number MC iterations, each consisting of self.nb_steps mutations
        """
        new_best_list = np.zeros(nb_iterations, dtype=bool)
        if self.stepwise_intermediates is not None:
            with open(self.stepwise_intermediates, 'w') as fh:
                fh.write(f'NUMMDL    {str(nb_iterations+1).ljust(4)}\n')
        if self.stepwise_intermediates is not None:
            pdb_str = ''.join(
                [f'MODEL     {str(0).ljust(4)}\n', self.best_model.get_pdb_coords(intermediate=True), 'ENDMDL\n'])
            with open(self.stepwise_intermediates, 'a') as fh: fh.write(pdb_str)

        for i in range(nb_iterations):
            movable_bool = self.candidate_model.apply_n_steps(self.nb_steps)
            if not movable_bool:
                print('No further moves possible')
                break
            if self.store_rg:
                self.rg_list.append(self.best_model.rg)
            if self.accept_or_reject():
                new_best_list[i] = True
                if not silent:
                    print(f'New best model iteration {i}, E: {self.best_model.base_energy}')
            if self.stepwise_intermediates is not None:
                pdb_str = ''.join([f'MODEL     {str(i + 1).ljust(4)}\n',  self.best_model.get_pdb_coords(intermediate=True), 'ENDMDL\n'])
                with open(self.stepwise_intermediates, 'a') as fh: fh.write(pdb_str)
                self.e_mat_dict[str(i+1)] = self.best_model.e_matrix[0]
            # self.iter += 1
        if self.stepwise_intermediates is not None:
            with open(self.stepwise_intermediates, 'a') as fh:
                fh.write(self.best_model.get_pdb_coords(conect_only=True))
            self.e_mat_dict['new_best'] = new_best_list
            np.savez(self.stepwise_intermediates.replace('_intermediates.pdb', '_e_mat.npz'), **self.e_mat_dict)

    def make_snapshots_mp(self, nb_snapshots, nb_iterations, free_sampling, out_queue):
        snapshot_list = [deepcopy(self.best_model)]
        movable_bool = True
        for sidx in range(nb_snapshots):
            for nit in range(nb_iterations):
                movable_bool = self.candidate_model.apply_n_steps(self.nb_steps)
                if not movable_bool: break
                self.accept_or_reject(free_sampling=free_sampling)
            # e_list.append(self.best_model.base_energy.round(3))
            snapshot_list.append(deepcopy(self.best_model))
            if self.store_rg:
                self.rg_list.append(self.candidate_model.rg)
            if not movable_bool: break
        if self.store_rg:
            out_queue.put((snapshot_list, self.rg_list))
        else:
            out_queue.put(snapshot_list)
        return

    def make_snapshots(self, nb_snapshots, nb_iterations, rg_list, fn, free_sampling=False):
        """
        Continue simulation, make snapshots and save ensemble in pdb file
        """
        # snapshot_list = [deepcopy(self.best_model)]
        # movable_bool = True
        # e_list = [self.best_model.base_energy.round(3)]
        out_queue = mp.Queue()
        p = mp.Process(target=self.make_snapshots_mp, args=(nb_snapshots, nb_iterations,free_sampling, out_queue))
        p.start()
        while out_queue.empty(): sleep(0.01)
        if self.store_rg:
            snapshot_list, self.rg_list = out_queue.get()
        else:
            snapshot_list = out_queue.get()

        # for sidx in range(nb_snapshots):
        #     for nit in range(nb_iterations):
        #         movable_bool = self.candidate_model.apply_n_steps(self.nb_steps)
        #         if not movable_bool: break
        #         self.accept_or_reject(free_sampling=free_sampling)
        #     # e_list.append(self.best_model.base_energy.round(3))
        #     snapshot_list.append(deepcopy(self.best_model))
        #     if self.store_rg:
        #         self.rg_list.append(self.candidate_model.rg)
        #     if not movable_bool: break

        # Create ensemble pdb
        fret_fingerprint = []
        dist_fingerprint = []
        e_list = []
        coord_list = []
        for ssidx, ss in enumerate(snapshot_list):
            fret_fingerprint.append(ss.fret_fingerprint)
            dist_fingerprint.append(ss.dist_fingerprint)
            coord_list.append( ''.join([f'MODEL     {str(ssidx + 1).ljust(4)}\n',
                                        ss.get_pdb_coords(intermediate=True),
                                        'ENDMDL\n']))
            e_list.append(ss.base_energy)
        coord_list.append(snapshot_list[0].get_pdb_coords(conect_only=True))
        last_mod = snapshot_list[-1]

        pdb_list = [f'REMARK   0 BASE ENERGY {last_mod.base_energy}\n']
        if rg_list is not None:
            for rg in rg_list: pdb_list.append(f'REMARK   1 RG {rg}\n')
        pdb_list.append(f'REMARK   1 FINGERPRINT {fret_fingerprint}\n')
        pdb_list.append(f'REMARK   1 DIST_FINGERPRINT {dist_fingerprint}\n')
        pdb_list.append(f'REMARK   1 ENERGIES {e_list}\n')
        pdb_list.append(f'NUMMDL    {str(nb_snapshots+1).ljust(4)}\n')
        pdb_list.extend(coord_list)
        pdb_txt = ''.join(pdb_list)
        with open(fn, 'w') as fh: fh.write(pdb_txt)

        return self.candidate_model

    def initialize(self):
        self.best_model = self.lattice_class(**self.kwargs)
        self.candidate_model = self.lattice_class(**self.kwargs)
