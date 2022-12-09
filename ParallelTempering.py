import numpy as np
import random
import multiprocessing as mp
from math import ceil
from time import sleep
import pandas as pd
from copy import deepcopy

from LatticeModelComparison import LatticeModelComparison
from helpers import print_timestamp


class ParallelTempering(object):
    def __init__(self,  **kwargs):
        # --- identifiers ---
        self.pdb_id = kwargs['pdb_id']
        self.reactive_idx = kwargs['reactive_idx']
        self.model_nb = str(kwargs.get('model_nb', 'backup'))

        # --- IO ---

        # --- run params ---
        self.beta_list = kwargs['beta_list']
        self.lattice_type = kwargs['lattice_type']
        self.experimental_mode = kwargs['experimental_mode']
        self.accomodate_tags = kwargs.get('accomodate_tags', False)

        self.nb_temps = len(self.beta_list)

        # Initialize with same seed for all betas
        LMC_list = []
        base_lmc = LatticeModelComparison(mod_id=0, beta=self.beta_list[0], **kwargs)
        for mod_id, beta in enumerate(self.beta_list):
            LMC_list.append(deepcopy(base_lmc))
            LMC_list[-1].mod_id = mod_id
            LMC_list[-1].beta = beta
        # Alt: initialize separately for each chain
        # out_queue = mp.Queue()
        # processes = [mp.Process(target=self.get_new_lmc, args=(beta, kwargs, out_queue)) for beta in self.beta_list]
        # for p in processes: p.start()
        # while True:
        #     running = any(p.is_alive() for p in processes)
        #     while not out_queue.empty():
        #         LMC_list.append(out_queue.get())
        #     if not running:
        #         break
        self.LMC_list = LMC_list

        self.nb_steps = kwargs['nb_steps']
        self.nb_iters = kwargs['nb_iters']
        self.iters_per_tempswap = kwargs['iters_per_tempswap']
        self.nb_tempswaps = self.nb_iters // self.iters_per_tempswap
        self.nb_processes = kwargs['nb_processes']
        self.save_dir = kwargs.get('save_dir', None)
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        self.store_rg = kwargs['store_rg']
        self.rg_list = []
        self.snapshots = kwargs['snapshots']
        self.save_intermediate_structures = kwargs['save_intermediate_structures']
        self.free_sampling = kwargs['free_sampling']

        # # Params for temp autotune
        # self.tswap_count = 0
        # self.t0 = self.nb_tempswaps * 10 / self.nb_temps
        # self.v_inv = self.nb_temps / self.nb_tempswaps
        # self.tswap_freqs = {t: [0] * 10 for t in range(1, self.nb_temps)}

        self.store_energies = kwargs.get('store_energies', False)
        if self.store_energies:
            self.store_energies = f'{self.save_dir}{self.pdb_id}_ri{self.reactive_idx}_{self.model_nb}_energies.tsv'
            with open(self.store_energies, 'w') as fh:
                fh.write('step\te_aa\te_ss\te_sol\te_dsb\te_tag\trg\n')
                en_list = [str(en.__round__(3)) for en in self.LMC_list[0].best_model.individual_energies]
                fh.write('\t'.join([str(0)] + en_list + [str(self.LMC_list[0].best_model.rg)]) + '\n')

        # objects to save stats
        self.temp_df = pd.DataFrame(
            index=pd.MultiIndex.from_product([np.arange(self.nb_tempswaps), np.arange(self.nb_processes)],
                                             names=['tempswap', 'mod_id']), columns=['temp_id', 'temp', 'E'])

        self.tagged_resn = kwargs['tagged_resn']  # only used for accomodate tags bit
        self.tagged_resi = kwargs['tagged_resi']

    @property
    def save_intermediate_structures(self):
        return self._save_intermediate_structures

    @save_intermediate_structures.setter
    def save_intermediate_structures(self, save_bool):
        if save_bool:
            fn = f'{self.save_dir}{self.pdb_id}_ri{self.reactive_idx}_{self.model_nb}_intermediates.pdb'
            with open(fn, 'w') as fh:
                fh.write(f'NUMMDL    {str(self.nb_tempswaps).ljust(4)}\n')
            self.intermediate_structures_fn = fn
        self._save_intermediate_structures = save_bool

    # def get_new_lmc(self, beta, kwargs, out_queue):
    #     '''
    #     Generate new LMC object, required for parallel generation of objects
    #     '''
    #     np.random.seed()
    #     random.seed()
    #     lmc = LatticeModelComparison(beta=beta, **kwargs)
    #     out_queue.put(lmc)

    def do_mc(self, LMC_idx, nb_iters, out_queue):
        np.random.seed()
        random.seed()
        self.LMC_list[LMC_idx].do_mc(nb_iters, silent=True)
        out_queue.put(self.LMC_list[LMC_idx])

    def do_mc_parallel(self):

        # structures to store e_mat stats
        new_best_list = []
        # new_best_list = np.zeros(self.nb_tempswaps, dtype=bool)
        e_mat_dict = {'0': self.LMC_list[0].best_model.e_matrix[0]}

        # Store starting structure
        if self.save_intermediate_structures:
            pdb_str = ''.join([f'MODEL     {str(0).ljust(4)}\n', self.LMC_list[0].best_model.get_pdb_coords(intermediate=True), 'ENDMDL\n'])
            with open(self.intermediate_structures_fn, 'a') as fh: fh.write(pdb_str)

        nb_proc_rounds = ceil(self.nb_temps / self.nb_processes)
        p_array = np.array_split(np.arange(self.nb_temps), nb_proc_rounds)  # if number of chains > nb processes, array denotes which processes to run as batch together
        nb_iters_list = [self.iters_per_tempswap] * self.nb_tempswaps + [self.nb_iters % self.iters_per_tempswap]  # list number of MC iters per tempswap round
        if nb_iters_list[-1] == 0: nb_iters_list = nb_iters_list[:-1]
        meta_best_model = None
        meta_best_e = np.inf
        meta_best_counter = 0
        # tsi = 0
        # if self.accomodate_tags:
        #     def prereq():
        #         return tsi < self.nb_tempswaps
        #         # return self.get_best_model()[1].individual_energies[4] != 0 or tsi < self.nb_tempswaps
        # else:
        #     def prereq():
        #         return tsi < self.nb_tempswaps
        # while prereq():
        for tsi in range(self.nb_tempswaps):
            LMC_cur_list = []
            for pa in p_array:
                out_queue = mp.Queue()
                # if self.accomodate_tags:
                #     processes = [mp.Process(target=self.do_mc, args=(i, self.iters_per_tempswap, out_queue)) for i in pa]
                # else:
                #     processes = [mp.Process(target=self.do_mc, args=(i, nb_iters_list[tsi], out_queue)) for i in pa]
                processes = [mp.Process(target=self.do_mc, args=(i, nb_iters_list[tsi], out_queue)) for i in pa]
                for p in processes: p.start()
                # for p in processes: p.join()
                # while not out_queue.empty(): LMC_cur_list.append(out_queue.get())
                while True:
                    sleep(0.0001)  # Reduce stalling due to repetitive checking of queue
                    running = any(p.is_alive() for p in processes)
                    while not out_queue.empty():
                        LMC_cur_list.append(out_queue.get())
                    if not running:
                        break

            self.LMC_list = LMC_cur_list
            self.LMC_list.sort(key=lambda x: x.beta, reverse=True)  # Sort LMC objects on beta value: models returned at random
            # if self.experimental_mode == 8:
            #     lowest_chain = self.LMC_list.pop(np.argwhere(np.isnan([lmc.beta for lmc in self.LMC_list]))[0,0])
            #     self.LMC_list = [lowest_chain] + self.LMC_list
            lmc_best, lm_best, e_best = self.get_best_model()


            # --- store temperature stats ---
            for lmc_id, lmc in enumerate(self.LMC_list):
                self.temp_df.loc[(tsi, lmc.mod_id), 'temp'] = 0.01 / lmc.beta
                self.temp_df.loc[(tsi, lmc.mod_id), 'temp_id'] = lmc_id
                self.temp_df.loc[(tsi, lmc.mod_id), 'E'] = lmc.best_model.base_energy

            # --- Print stats & save intermediate results ---
            print(f'{print_timestamp()} Temp round {tsi+1} - best E: {e_best}, at T: {0.01 / lmc_best.beta} Rg: {lm_best.rg}')

            # Save e matrix
            e_mat_dict[str(tsi+1)] = lm_best.e_matrix[0]

            # Save best overall model
            if e_best < meta_best_e:
                meta_best_model = deepcopy(lm_best)
                meta_best_e = e_best
                print(f'{print_timestamp()} New best model!')
                new_best_list.append(True)
                meta_best_counter = 0  # required for early stopping mechanism
            else:
                meta_best_counter += 1

            # save intermediate structure
            if self.save_intermediate_structures:
                pdb_str = ''.join([f'MODEL     {str(tsi + 1).ljust(4)}\n',  lm_best.get_pdb_coords(intermediate=True), 'ENDMDL\n'])
                with open(self.intermediate_structures_fn, 'a') as fh: fh.write(pdb_str)

            # Save Rg
            if self.store_rg == 'tswap':
                self.rg_list.append(lm_best.rg)
            elif self.store_rg == 'full':
                self.rg_list.extend(lmc_best.rg_list)
                for li in range(len(self.LMC_list)): self.LMC_list[li].rg_list = []

            # Save energies
            if self.store_energies:
                with open(self.store_energies, 'a') as fh:
                    en_list = [str(en.__round__(3)) for en in lm_best.individual_energies]
                    fh.write('\t'.join([str(tsi+1)] + en_list + [str(lm_best.rg)]) + '\n')

            # --- Early stopping ---
            if self.early_stopping_bool and meta_best_counter > self.early_stopping_rounds:
                break

            # --- exchange models between chains ---
            self.exchange_temps()
            # tsi += 1

        lmc = LatticeModelComparison(mod_id=0, lm=meta_best_model, beta=max(self.beta_list), nb_steps=1,
                                     store_rg=self.store_rg,
                                     lattice_type=self.lattice_type)

        if self.accomodate_tags:
            if self.get_best_model()[1].individual_energies[4] != 0:
                np.savez(f'{self.save_dir}{self.pdb_id}_ri{self.reactive_idx}_{self.model_nb}_unoptimizedTags.npz',
                         sequence=lmc.best_model.aa_sequence, coords=lmc.best_model.coords,
                         secondary_structure=lmc.best_model.ss_df.to_numpy(),
                         tagged_resi=self.tagged_resi)
                tagged_resn = 'None' if self.tagged_resn == '' else self.tagged_resn
                with open(f'{self.save_dir}../../unfinished_ids.txt', 'a') as fh: fh.write(f'{self.pdb_id}\t{self.model_nb}\t{tagged_resn}\n')
                raise ValueError('Could not find model accomodating all tags')


        if self.save_intermediate_structures:
            with open(self.intermediate_structures_fn, 'a') as fh:
                fh.write(lmc.best_model.get_pdb_coords(conect_only=True))

        # --- Save temperature stats ---
        self.temp_df.reset_index(inplace=True)
        self.temp_df.to_csv(f'{self.save_dir}{self.pdb_id}_ri{self.reactive_idx}_{self.model_nb}_tempstats.tsv',
                            sep='\t', header=True, index=False)

        # --- Save e-matrices ---
        e_mat_dict['sequence'] = self.LMC_list[0].best_model.aa_sequence
        e_mat_dict['new_best'] = np.array(new_best_list)
        np.savez(f'{self.save_dir}{self.pdb_id}_ri{self.reactive_idx}_{self.model_nb}_e_mat.npz', **e_mat_dict)

        # --- Save coordinates as npz ---
        np.savez(f'{self.save_dir}{self.pdb_id}_ri{self.reactive_idx}_{self.model_nb}.npz',
                 sequence=lmc.best_model.aa_sequence, coords=lmc.best_model.coords,
                 secondary_structure=lmc.best_model.ss_df.to_numpy())

        # --- Save snapshots ---
        # if self.snapshots[0] < 1: return
        print(f'Optimization done! Making {self.snapshots[0]} snapshots...')
        lmc.make_snapshots(self.snapshots[0], self.snapshots[1], self.rg_list,
                           f'{self.save_dir}{self.pdb_id}_ri{self.reactive_idx}_{self.model_nb}.pdb',
                           free_sampling=self.free_sampling)

    def get_p_acc(self, idx):
        """
        return probability of accepting swap of models between temperature chains
        """
        delta_e = self.LMC_list[idx[0]].best_model.base_energy - self.LMC_list[idx[1]].best_model.base_energy
        if np.isnan(self.LMC_list[idx[0]].beta):
            return 0 if delta_e < 1 else 1
        delta_b = self.LMC_list[idx[0]].beta - self.LMC_list[idx[1]].beta
        p_acc = np.exp(min(709.7, delta_e * delta_b))  # min avoids overflow warning
        return min(p_acc, 1)

    def exchange_temps(self):
        """
        Exchange temperatures between chains based on their differences in energy and temperature
        """
        ai = {n: 0 for n in range(1, self.nb_temps)}
        exchange_dict = dict()
        for il in range(1, self.nb_temps):
            if random.uniform(0, 1) < self.get_p_acc((il-1, il)):
                exchange_dict[il] = il-1
                ai[il] = 1

        # Exchange temps and sort models
        for ed in exchange_dict:
            self.LMC_list[ed].beta, self.LMC_list[exchange_dict[ed]].beta = \
                self.LMC_list[exchange_dict[ed]].beta, self.LMC_list[ed].beta
        # self.LMC_list.sort(key=lambda x: x.beta)
        # # autotune temperatures
        # self.tswap_count += 1
        # kt = self.v_inv * (self.t0 / (self.t0 + self.tswap_count))
        # si_list = list()
        # for il in range(1, self.nb_temps-1):
        #     si = kt * (ai[il] - ai[il + 1]) + log(self.beta_list[il - 1]- self.beta_list[il])
        #     si_list.append(si)
        # for il in range(self.nb_temps-2):
        #     self.LMC_list[il + 1].beta = exp(si_list[il]) + self.LMC_list[il].beta
        self.LMC_list.sort(key=lambda x: x.beta, reverse=True)
        self.beta_list = [lmc.beta for lmc in self.LMC_list]
        # for aii in ai: self.tswap_freqs[aii] = self.tswap_freqs[aii][1:] + [ai[aii]]

    def get_best_model(self):
        """
        Return tuple: LMC containing lowest-E model, lowest-E model and its lowest E value
        """
        e_list = np.array([lmc.best_model.base_energy for lmc in self.LMC_list])
        return self.LMC_list[np.argmin(e_list)], self.LMC_list[np.argmin(e_list)].best_model, np.min(e_list)

    @property
    def early_stopping_rounds(self):
        return self._early_stopping_rounds

    @early_stopping_rounds.setter
    def early_stopping_rounds(self, esr):
        self._early_stopping_rounds = esr
        if esr < -1 or esr == 0:
            raise ValueError(f'{esr} is not a valid number of early stopping rounds, supply a positive integer'
                             f'to signify number of early stopping rounds or -1 to disable.')
        elif esr == -1:
            self.early_stopping_bool = False
        else:
            self.early_stopping_bool = True
