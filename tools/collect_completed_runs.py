import argparse, os, sys, re
from os.path import basename, splitext
from shutil import copyfile
import numpy as np
from itertools import chain
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])

from helpers import parse_input_dir, parse_output_dir

parser = argparse.ArgumentParser(description='Collect lattice structures that were completed for all labeling schemes')
parser.add_argument('--in-dirs', required=True, nargs='+', type=str)
parser.add_argument('--out-dir', required=True, type=str)
parser.add_argument('--success-txt', required=True, type=str,
                    help='output file for ID list of successfully modeled proteins')
parser.add_argument('--max-fold', required=False, type=int,
                    help='Number of models to expect, if less: leave protein out. If more: subsample (models are renamed)')
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)

all_tag_dict = {}

# collect step
for in_dir in args.in_dirs:
    tag_cat = re.search('tag[A-Z]+', in_dir).group(0)
    pdb_list = parse_input_dir(in_dir, pattern='*.pdb', regex='[0-9].pdb$')  # removes intermediates files with regex
    pdb_dict = {}
    for pdb_fn in pdb_list:
        struct_cat, struct_id = pdb_fn.split('/')[-2:]
        pdb_dict[struct_cat] = pdb_dict.get(struct_cat, []) + [pdb_fn]
    all_tag_dict[tag_cat] = pdb_dict

# retain structures with at least max_fold representatives for each labeling scheme
struct_types_per_tag_cat = [{tt2:len(all_tag_dict[tt][tt2]) for tt2 in all_tag_dict[tt]} for tt in all_tag_dict]
struct_types_list = set(list(chain.from_iterable(struct_types_per_tag_cat)))
st_retain = []
if args.max_fold:
    for stl in struct_types_list:
        if all([tcl.get(stl, 0) >= args.max_fold for tcl in struct_types_per_tag_cat]): st_retain.append(stl)
else:
    st_retain = struct_types_list
all_tag_dict = {t: {t2: all_tag_dict[t][t2] for t2 in all_tag_dict[t] if t2 in st_retain} for t in all_tag_dict}
with open(args.success_txt, 'w') as fh: fh.write('\n'.join(list(all_tag_dict[list(all_tag_dict)[0]])))

# Store selected structures in new place
for t in all_tag_dict:
    od = parse_output_dir(out_dir + t)
    for s in all_tag_dict[t]:
        if args.max_fold:
            struct_out_list = np.sort(np.random.choice(all_tag_dict[t][s], args.max_fold, replace=False))
        else:
            struct_out_list = all_tag_dict[t][s]
        for idx, fn in enumerate(struct_out_list):
            new_fn = re.search('.+(?=[0-9]+.pdb)', basename(fn)).group(0)
            copyfile(fn, f'{od}{new_fn}{idx}.pdb')
