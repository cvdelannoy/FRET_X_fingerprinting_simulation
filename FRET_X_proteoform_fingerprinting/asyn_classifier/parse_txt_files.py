import argparse, sys, re, pickle
import numpy as np

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from helpers import parse_input_dir, parse_output_dir

class ExperimentData(object):
    def __init__(self, txt_fn, exp_id, cutoff_low, cutoff_high, fp_len, check_donor_only):
        self.data_dict = {'X': {'nb_tags': 0, 'fret': [], 'dist': [], 'fingerprint': [], 'coords': []}}
        self.fret_list = []
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.fp_len = fp_len
        self.check_donor_only = check_donor_only

        self.exp_id = exp_id
        self.txt_fn = txt_fn

        self.parse_txt_file()

    @property
    def nb_molecules(self):
        return len(self.data_dict['X']['fret'])


    @property
    def fret_resolution(self):
        return self._fret_resolution

    @fret_resolution.setter
    def fret_resolution(self, fr):
        self._fret_resolution = fr
        self._res_bins = np.arange(0.01, 1 + args.fret_resolution, args.fret_resolution)

    def parse_txt_file(self):

        with open(self.txt_fn, 'r') as fh:
            txt_str = fh.read()
        for mol_txt in txt_str.split('\n\n')[1:]:
            self.add_mol(mol_txt)

    def add_mol(self, mt):
        mt_list = mt.split('\n')
        mol_idx = int(re.search('(?<=Molecules )[0-9]+', mt_list.pop(0)).group(0))
        event_dict = {}
        for x in mt_list:
            x = x.strip()
            if not len(x): continue
            if 'No barcode found' in x: return
            if x.startswith('Barcode'):
                barcode_idx = int(x.split(' ')[1][:-1])
                event_dict[barcode_idx] = {}
            else:
                var_name, value = x.rsplit(' ', 1)
                value = float(value)
                if value > self.cutoff_high: continue
                event_dict[barcode_idx][var_name] = value
        event_dict = {edi: event_dict[edi] for edi in event_dict if 'position' in event_dict[edi]}
        if self.check_donor_only:
            # check for donor-only peak
            donor_only_idx = np.squeeze(np.argwhere([event_dict[i]['position'] < self.cutoff_low for i in event_dict]))
            if donor_only_idx.size > 1:
                return
            elif donor_only_idx.size == 1:
                event_dict = {edi: event_dict[edi] for edi in event_dict if edi != donor_only_idx}
        if len(event_dict) != self.fp_len:
            return
        self.fret_list.append([event_dict[edi]['position'] for edi in event_dict][0])

    def save(self, out_dir):
        fret_dict = {
            'up_id': self.exp_id,
            'anchor_type': 'other',
            'efficiency': 1.0,
            'nb_examples': self.nb_molecules,
            'data': self.data_dict
        }
        with open(f'{out_dir}{self.exp_id}.pkl', 'wb') as fh:
            pickle.dump(fret_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser(description='Parse FRET values from SHKs text format')
parser.add_argument('--in-dir', type=str, required=True)
parser.add_argument('--fp-len', type=int, default=1)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--cutoffs', type=float, nargs=2, default=[0.1, 0.9])
parser.add_argument('--out-fmt', type=str, choices=['pkl', 'txt'], default='txt')
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)
txt_list = parse_input_dir(args.in_dir, pattern='*.txt')

for txt_fn in txt_list:
    exp_id = Path(txt_fn).stem
    exp_data = ExperimentData(txt_fn, exp_id, args.cutoffs[0], args.cutoffs[1], args.fp_len, True)
    if args.out_fmt == 'pkl':
        exp_data.save(out_dir)
    elif args.out_fmt == 'txt':
        with open(out_dir + exp_data.exp_id + '.txt', 'w') as fh:
            fh.write('\n'.join([str(x) for x in exp_data.fret_list]))
