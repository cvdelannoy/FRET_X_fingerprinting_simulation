import argparse, os, sys, re
import numpy as np

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from helpers import parse_input_dir, parse_output_dir

dummy_fn = str(Path(__file__).resolve().parent) + '/dummy_zeros.txt'

parser = argparse.ArgumentParser(description='Randomly combine N- and C-terminus FRET values to artificial 2-label molecules')
parser.add_argument('--n-term', type=str, required=True)
parser.add_argument('--c-term', type=str, required=True)
parser.add_argument('--regex', type=str, default='[A-Z][0-9]+[A-Z](?=.txt)')
parser.add_argument('--out-dir',type=str, required=True)
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)

nterm_fn_dict = {re.search(args.regex, fn).group(0): fn for fn in  parse_input_dir(args.n_term, '*.txt')}
cterm_fn_dict = {re.search(args.regex, fn).group(0): fn for fn in  parse_input_dir(args.c_term, '*.txt')}

all_ids = set(list(nterm_fn_dict) + list(cterm_fn_dict))

for cur_id in all_ids:
    nterm_array = np.loadtxt(nterm_fn_dict.get(cur_id, dummy_fn))
    cterm_array = np.loadtxt(cterm_fn_dict.get(cur_id, dummy_fn))

    nb_vals = min(nterm_array.size, cterm_array.size)

    np.random.shuffle(nterm_array)
    np.random.shuffle(cterm_array)
    out_array = np.vstack((nterm_array[:nb_vals], cterm_array[:nb_vals])).T
    np.savetxt(f'{out_dir}CN_term_{cur_id}', out_array)
    print(f'processed {cur_id}...')
print('Done')





