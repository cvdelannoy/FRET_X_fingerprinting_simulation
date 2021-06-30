import argparse, os, sys, re
from os.path import splitext, basename
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import ast
import numpy as np
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
import helpers as nhp

def str2list(line, id_str):
    return np.array(ast.literal_eval(line.replace(id_str, '')))
    # if levels == 1:
    #     return np.array([float(n.strip('[]')) for n in line.replace(id_str, '').strip('[]\n').split(', ')])
    # if levels == 2:
    #     # return [[float(nn) for nn in n.split(', ')] for n in line.replace(id_str, '').strip('[]\n').split('], [')]
    #     fp = [[nn for nn in n.strip('[]\n').split('], [')] for n in
    #           line.replace(id_str, '').strip('[]\n').split(', ')]
    #     fp2 = []
    #     for ni, n in enumerate(fp):
    #         sub_fp=[]
    #         for nn in n:
    #             if len(nn):
    #                 val = float(nn)
    #             else:
    #                 val = None
    #             sub_fp.append(val)
    #         fp2.append(sub_fp)
    #     return fp2


parser = argparse.ArgumentParser(description='Takes pdb files and produces assessment of separability based on '
                                             'FRETpaint method.')
parser.add_argument('--wd', type=str, required=True,
                    help='working directory')
parser.add_argument('--efret-res', type=float, default=0.01,
                    help='Resolution in FRET efficiency, i.e. minimum difference still considered discernible.')
parser.add_argument('--nb-tags', type=int, default=1,
                    help='Number of tags to separate.')
parser.add_argument('--dist-res', type=float, default=1.0,
                    help='Resolution in distance, i.e. the minimum donor/acceptor difference still considered discernible.')
parser.add_argument('--detection-limit', type=float, default=0.10,
                    help='Minimum fret efficiency still considered readable.')
parser.add_argument('--pdb-dir', type=str, required=True, nargs='+',
                    help='directory containing lattice model pdb files.')
parser.add_argument('--pattern', type=str, default='^.+(?=_[0-9]+)',
                   help='Treat all names that adhere to the same regex pattern as different models of the same'
                        'peptide.')
parser.add_argument('--event-duration', type=int, default=-1,
                    help='Define how many snapshots go in 1 event. provide -1 to get 1 event per structure [default: -1]')
parser.add_argument('--max-events', type=int, default=np.inf,
                    help='Maximum number of events per molecule to take. [default: infinite]')
parser.add_argument('--observed-tags', action='store_true', help='Compare based on nb seen tags i.o. nb present.')
parser.add_argument('--exp-data', type=str,
                    help='Folder(s)/files containing experimental Efret values, with same names as pdbs')
parser.add_argument('--skip-expensive-plots', action='store_true')

args = parser.parse_args()

pdb_list = nhp.parse_input_dir(args.pdb_dir, '*.pdb')
wd = nhp.parse_output_dir(args.wd, clean=True)
fp_vs_time_dir = nhp.parse_output_dir(wd+'fp_vs_time')

fp_dict = {}
fp_dist_dict = {}
e_dict = {}

for pdb_fn in pdb_list:
    pdb_id = splitext(basename(pdb_fn))[0]
    try:
        with open(pdb_fn, 'r') as fh:
            efret_fp, dist_fp, energy_fp = None, None, None
            while efret_fp is None or dist_fp is None or energy_fp is None:
                line = fh.readline()
                if '1 FINGERPRINT' in line:
                    fp_obj = str2list(line, 'REMARK   1 FINGERPRINT ')
                    if len(fp_obj):
                        if type(fp_obj[0]) == dict:
                            efret_fp = np.array([x.get('C', []) for x in fp_obj])
                        else:  # list
                            efret_fp = np.array(fp_obj)
                elif '1 DIST_FINGERPRINT' in line:
                    fp_obj = str2list(line, 'REMARK   1 DIST_FINGERPRINT ')
                    if len(fp_obj):
                        if type(fp_obj[0]) == dict:
                            dist_fp = np.array([x.get('C', []) for x in fp_obj])
                        else:  # list
                            dist_fp = np.array(fp_obj)
                elif '1 ENERGIES' in line:
                    energy_fp = str2list(line, 'REMARK   1 ENERGIES ')
        keep_bool = [len(efp) == args.nb_tags for efp in efret_fp]
        efret_fp, dist_fp, energy_fp = np.vstack(efret_fp[keep_bool]), np.vstack(dist_fp[keep_bool]), np.hstack(energy_fp[keep_bool])
        if args.event_duration == -1:
            nb_events, event_duration = 1, len(energy_fp)
        else:
            nb_events, event_duration = min(args.max_events, len(energy_fp) // args.event_duration), args.event_duration

        for nt in range(args.nb_tags):
            for event_idx in range(nb_events):
                ll = event_idx * event_duration
                rl = ll + event_duration
                (fp_dict[f'{pdb_id}_tag{nt}_event{event_idx}'],
                 fp_dist_dict[f'{pdb_id}_tag{nt}_event{event_idx}'],
                 e_dict[f'{pdb_id}_tag{nt}_event{event_idx}']) = efret_fp[ll:rl,nt], dist_fp[ll:rl,nt], energy_fp[ll:rl]
        if not args.skip_expensive_plots:
            plt.plot(efret_fp)
            plt.savefig(f'{fp_vs_time_dir}{pdb_id}.svg')
            plt.close(plt.gcf())
    except Exception as e:
        print(f'{pdb_id} fucked up with reason {e}')
        continue

# --- plot per-pdb fingerprints ---
df_list = []
for pdb_id in fp_dict:
    c_type = int(re.search('(?<=C)[0-9]+', pdb_id).group(0))
    df_list.append(pd.DataFrame({'$E_{FRET}$': fp_dict[pdb_id],
                                 'pdb_id': [pdb_id] * len(fp_dict[pdb_id]),
                                 'energy': e_dict[pdb_id],
                                 'c_type': c_type}))
fp_df = pd.concat(df_list)
fp_df.loc[:, '$E_{FRET}$'] = fp_df.loc[:, '$E_{FRET}$'].astype(float)

if not args.skip_expensive_plots:
    for ct, ctdf in fp_df.groupby('c_type'):
        fig=plt.figure(figsize=(15,5))
        violin_axis = sns.violinplot(x='pdb_id', y='$E_{FRET}$', scale='width', inner='stick',
                                     data=ctdf, split=True, cut=0)
        plt.savefig(f'{wd}per_pdb_violin_C{ct}.svg', dpi=400)
        plt.close(fig)

for ct, ctdf in fp_df.groupby('c_type'):
    fig=plt.figure(figsize=(10,10))
    sns.scatterplot(x='energy', y='$E_{FRET}$', data=ctdf)
    plt.savefig(f'{wd}per_pdb_scatter_C{ct}.png', dpi=400)
    plt.close(fig)

# --- plot per-structure fingerprints ---
fp_df.loc[:, 'group_id'] = fp_df.pdb_id.apply(lambda x: re.search(args.pattern, x).group(0) + '_' +
                                              re.search('tag[0-9]+', x).group(0))

fig=plt.figure(figsize=(15,5))
violin_axis = sns.violinplot(x='group_id', y='$E_{FRET}$', scale='width', inner='stick',
                             data=fp_df, split=True, cut=0)
plt.savefig(f'{wd}per_group_violin.svg', dpi=400)

fp_avg_df = pd.DataFrame(columns=['$E_{FRET}$', 'sd', 'mod_id', 'group_id'])
for idf, ddf in fp_df.groupby('pdb_id'):
    fp_avg_df.loc[idf, :] = [np.sum(ddf.loc[:, "$E_{FRET}$"] * (ddf.energy / ddf.energy.sum())),
                             ddf.loc[:, "$E_{FRET}$"].std(), idf, ddf.group_id.to_list()[0]]

def plot_group_beeswarm(df, var, fn, exp_df):
    fig = plt.figure(figsize=(15, 5))
    sns.boxplot(x='group_id', y=var, color='white', data=df)
    swarm_ax = sns.swarmplot(x='group_id', y=var, data=df)

    if exp_df is not None:
        for lid, tup in exp_df.iterrows():
            swarm_ax.axhline(tup.loc[var], tup.lb, tup.rb, color='red')

    plt.savefig(fn, dpi=400)
    plt.close(fig)

fp_avg_df.loc[:, 'distance'] = fp_avg_df.loc[:,'$E_{FRET}$'].apply(nhp.get_FRET_distance)
if args.exp_data:
    exp_df = pd.read_csv(args.exp_data, index_col=0, header=0)
    exp_df.loc[:, 'distance'] = exp_df.efret.apply(nhp.get_FRET_distance)
    exp_df.rename({'efret': '$E_{FRET}$'}, inplace=True, axis=1)
    sections = np.linspace(0, 1, len(exp_df) + 1)
    exp_df.loc[:, 'lb'] = sections[:-1]
    exp_df.loc[:, 'rb'] = sections[1:]
else:
    exp_df = None
plot_group_beeswarm(fp_avg_df, '$E_{FRET}$', f'{wd}per_group_beeswarm_efret.svg', exp_df)
plot_group_beeswarm(fp_avg_df, 'distance', f'{wd}per_group_beeswarm_dist.svg', exp_df)
#
#
# fig=plt.figure(figsize=(15,5))
# if args.translate_to_dist:
#     fp_avg_df.loc[:, 'distance'] = fp_avg_df.apply(lambda x: nhp.get_FRET_distance(x.loc['$E_{FRET}$']), axis=1)
#     sns.boxplot(x='group_id', y='distance', color='white', data=fp_avg_df)
#     swarm_ax = sns.swarmplot(x='group_id', y='distance', data=fp_avg_df)
# else:
#     sns.boxplot(x='group_id', y='$E_{FRET}$', color='white', data=fp_avg_df)
#     swarm_ax = sns.swarmplot(x='group_id', y='$E_{FRET}$', data=fp_avg_df)
#
# if args.exp_data:
#     exp_df = pd.read_csv(args.exp_data, index_col=0, header=0)
#     sections = np.linspace(0, 1, len(exp_df) + 1)
#     exp_df.loc[:, 'lb'] = sections[:-1]
#     exp_df.loc[:, 'rb'] = sections[1:]
#     if args.translate_to_dist:
#         exp_df.efret = exp_df.apply(lambda x: nhp.get_FRET_distance(x.efret), axis=1)
#     for lid, tup in exp_df.iterrows():
#         swarm_ax.axhline(tup.efret, tup.lb, tup.rb, color='red')
#
# plt.savefig(f'{wd}per_group_beeswarm.svg', dpi=400)

fp_avg_df.to_csv(f'{wd}model40_comparison_avg.csv', index=False)
fp_df.to_csv(f'{wd}model40_comparison.csv', index=False)
