import argparse, sys, os, pickle
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])
from helpers import parse_output_dir

color_dict = {'C':'#66c2a5',
              'K': '#fc8d62',
              'R': '#8da0cb'}

parser = argparse.ArgumentParser(description='Graph fingerprints from a pickled set of pdbs')
parser.add_argument('--fp-pkl', required=True, type=str,
                    help='pickled fingerprints as produced by parse_fingerprints.py')
parser.add_argument('--out-dir',required=True, type=str)
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)
with open(args.fp_pkl, 'rb') as fh: struct_dict = pickle.load(fh)
for pdb_id in struct_dict:
    nb_fp = len(struct_dict[pdb_id]['fingerprints'])
    # fig, ax_list = plt.subplots(nb_fp, 1, figsize=(10,5 * nb_fp))

    array_list = []
    color_list = []
    lens_list = []
    ls_list = []
    for fpi in struct_dict[pdb_id]['fingerprints']:
        fp_dict = struct_dict[pdb_id]['fingerprints'][fpi]
        fp = []
        segments = []
        colors = []
        lens = []
        for ri, resn in enumerate(fp_dict):
            segments = [[(it+ri*0.3, fpi), (it+ri*0.3, fpi+fs*0.5)] for it, fs in enumerate(fp_dict[resn]) if fs != 0]
            fp_cur = np.argwhere(fp_dict[resn] != 0).reshape(-1)
            colors.extend(len(fp_cur) * [color_dict[resn]])
            lens.extend(fp_dict[resn])
            fp.append(fp_cur)
            ls_list.append(LineCollection(segments, colors=len(segments) * [color_dict[resn]], linewidths=2))
        fp = np.concatenate(fp)
        array_list.append(fp)
        color_list.append(colors)
        lens_list.append(lens)
    # plt.figure(figsize=(10, 5))
    fig,ax = pl.subplots(figsize=(15,10))
    for ls in ls_list: ax.add_collection(ls)
    ax.autoscale()
    # plt.eventplot(array_list, colors=color_list, linelengths=lens_list)
    id_list = list(struct_dict[pdb_id]['fingerprints'])
    plt.yticks(list(range(len(id_list))), id_list)
    plt.xlabel('FRET (E)'); plt.ylabel('fingerprint #')
    plt.savefig(f'{out_dir}{pdb_id}.svg')
    np.savetxt(f'{out_dir}{pdb_id}.tsv', np.array(lens_list), delimiter='\t')
    plt.close(fig=plt.gcf())

# plt.eventplot(frag_lens)
# plt.yticks(list(range(len(id_list))), id_list)
# plt.xlabel('Weight (Da)')
