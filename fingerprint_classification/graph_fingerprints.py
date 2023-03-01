import argparse, sys, os, pickle
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])
from helpers import parse_output_dir

color_dict = {'C':'#66c2a5',
              'K': '#fc8d62',
              'M': '#7fc97f',
               'Y': '#386cb0',
              'R': '#8da0cb'}

color_dict = { 'K': ['#756bb1', '#bcbddc'],
              'C': ['#e34a33', '#fdbb84'],
               'Y': ['#386cb0', '#bdc9e1'],
               'M': ['#7fc97f', '#31b031']}


parser = argparse.ArgumentParser(description='Graph fingerprints from a pickled set of pdbs')
parser.add_argument('--fp-pkl', required=True, type=str,
                    help='pickled fingerprints as produced by parse_fingerprints.py')
parser.add_argument('--xy-fp', action='store_true')
parser.add_argument('--out-dir',required=True, type=str)
args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir)
with open(args.fp_pkl, 'rb') as fh: struct_dict = pickle.load(fh)
for pdb_id in struct_dict:
    nb_fp = len(struct_dict[pdb_id]['fingerprints'])
    # fig, ax_list = plt.subplots(nb_fp, 1, figsize=(10,5 * nb_fp))
    legend_dict = {}
    array_list = []
    # color_list = []
    lens_list = []
    ls_list = []
    for fpi in struct_dict[pdb_id]['fingerprints']:
        fp_dict = struct_dict[pdb_id]['fingerprints'][fpi]
        fp = []
        segments = []
        # colors = []
        lens = []
        for rei in fp_dict:
            for ri, resn in enumerate(fp_dict[rei]):
                if rei == 1 and resn == 'C':
                    cp=1
                if args.xy_fp:
                    segments = [[(it+ri*0.3, fpi+rei*0.25), (it+ri*0.3, fpi+(rei+1)*0.25)]
                                for it, fs in enumerate(fp_dict[rei][resn]) if fs != 0]
                    fp_cur = np.argwhere(fp_dict[rei][resn] != 0).reshape(-1)
                    # colors.extend(len(fp_cur) * [color_dict[resn][0]])
                else:
                    segments = [[(it+ri*0.3, fpi), (it+ri*0.3, fpi+fs*0.5)] for it, fs in enumerate(fp_dict[rei][resn]) if fs != 0]
                    fp_cur = np.argwhere(fp_dict[rei][resn] != 0).reshape(-1)
                    # colors.extend(len(fp_cur) * [color_dict[resn][rei]])
                lens.extend(fp_dict[rei][resn])
                fp.append(fp_cur)
                ls_list.append(LineCollection(segments, colors=len(segments) * [color_dict[resn][rei]], linewidths=2))
                legend_key = f'{resn}, {["low pKa", "high pKa"][rei]}'
                if len(segments) and legend_key not in legend_dict:
                    legend_dict[legend_key] = Patch(facecolor=color_dict[resn][rei], label=legend_key)
        fp = np.concatenate(fp)
        array_list.append(fp)
        # color_list.append(colors)
        lens_list.append(lens)
    # plt.figure(figsize=(10, 5))
    fig,ax = pl.subplots(figsize=(10,6))
    for ls in ls_list: ax.add_collection(ls)
    ax.autoscale()
    # plt.eventplot(array_list, colors=color_list, linelengths=lens_list)
    id_list = list(struct_dict[pdb_id]['fingerprints'])
    plt.xlim([0, 100])
    plt.yticks(list(range(len(id_list))), id_list)
    plt.xlabel('FRET (E)'); plt.ylabel('fingerprint #')
    ax.legend(handles=list(legend_dict.values()), bbox_to_anchor=(0.5,-0.1), loc="upper center")
    plt.tight_layout()
    plt.savefig(f'{out_dir}{pdb_id}.svg')
    np.savetxt(f'{out_dir}{pdb_id}.tsv', np.array(lens_list), delimiter='\t')
    plt.close(fig=plt.gcf())

# plt.eventplot(frag_lens)
# plt.yticks(list(range(len(id_list))), id_list)
# plt.xlabel('Weight (Da)')
