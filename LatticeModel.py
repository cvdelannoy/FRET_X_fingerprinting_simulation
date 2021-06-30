import numpy as np
from itertools import chain
import helpers as nhp
from abc import ABC, abstractmethod, abstractproperty
from collections import ChainMap


class LatticeModel(ABC):
    def __init__(self, **kwargs):
        self.starting_structure = kwargs.get('starting_structure', 'random')
        self.aa_sequence = np.array(list(kwargs['aa_sequence']), dtype='<U3')
        self.aa_sequence_original = self.aa_sequence.copy()
        self.tagged_resi = kwargs['tagged_resi']
        self.seq_length = self.aa_sequence.size
        self.ss_sequence = kwargs['secondary_structure']

    @property
    def ss_sequence(self):
        return self._ss_sequence

    @ss_sequence.setter
    def ss_sequence(self, ss_df):
        ss_array = ss_df.apply(lambda x: ['H', 'S', 'L'][x.argmin()], axis=1).to_numpy().astype('<U1')
        ss_array[ss_df.apply(lambda x: np.all(x == x[0]), axis=1)] = 'L'  # If all probabilities equal: call it loop
        h_runoff = 0
        for si, ss in enumerate(ss_array):
            if ss == 'H':
                h_runoff = 4
            else:
                if h_runoff != 0:
                    ss_array[si] = 'H'
                    h_runoff -= 1
        self._ss_sequence = ss_array

    @property
    @abstractmethod
    def base_energy(self):
        pass

    @property
    @abstractmethod
    def rg(self):
        pass

    @property
    def tagged_resi(self):
        return self._tagged_resi

    @tagged_resi.setter
    def tagged_resi(self, tagged_dict):
        if not len(tagged_dict):
            self._tagged_resi = []
            return
        inv_list = [{v: k for v in tagged_dict[k]} for k in tagged_dict]
        inv_dict = dict(ChainMap(*inv_list))
        self.tagged_resi_dict = inv_dict
        self._tagged_resi = list(chain.from_iterable(tagged_dict.values()))
        if 0 not in self._tagged_resi:
            self._tagged_resi.insert(0,0)
        self.aa_sequence[self._tagged_resi] = 'TAG'


    @abstractmethod
    def apply_n_steps(self, n):
        pass

    @property
    @abstractmethod
    def coords(self):
        """
        CA coords
        """
        pass

    @coords.setter
    @abstractmethod
    def coords(self, new_coords):
        pass



        # --- IO methods ---

    @property
    def fret_fingerprint(self):
        """
        Return derived FRET values of tagged residues as list of floats
        """
        dist_fp = self.dist_fingerprint
        return {resn: [nhp.get_FRET_efficiency(dist) for dist in dist_fp[resn]] for resn in dist_fp}

    @property
    @abstractmethod
    def dist_fingerprint(self):
        pass

    @abstractmethod
    def get_pdb_coords(self):
        pass

