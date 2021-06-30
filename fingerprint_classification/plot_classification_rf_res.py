import os, sys, argparse, re
import pandas as pd
import numpy as np
from os.path import basename, splitext, dirname
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])

from helpers import parse_input_dir, aa_dict

mpl.rc('font', **{'size': 16})

color_dict = {'C':'#66c2a5',
              'CK': '#fc8d62',
              'KR': '#e78ac3',
              'CKR': '#8da0cb'}

sort_key = {('Perfect', 'KR'):0,  ('Perfect', 'CKR'): 1, ('Perfect', 'CK'): 2, ('Perfect', 'C'): 3,
            ('Suboptimal', 'CKR'): 4, ('Suboptimal', 'CK'): 5, ('Suboptimal', 'C'): 6}
#
# color_dict = {'C':'#66c2a5',
#               'CK': '#fc8d62',
#               'CKR': '#8da0cb'}
#
# sort_key = {('Perfect', 'CKR'): 0, ('Perfect', 'CK'): 1, ('Perfect', 'C'): 2,
#             ('Suboptimal', 'CKR'): 3, ('Suboptimal', 'CK'): 4, ('Suboptimal', 'C'): 5}

parser = argparse.ArgumentParser(description='Plot results rf classification results per resolution level')
parser.add_argument('--result-csv', type=str, nargs='+', required=True)
parser.add_argument('--out-svg', type=str, required=True)
args = parser.parse_args()

out_file_base = splitext(args.out_svg)[0]
csv_list = parse_input_dir(args.result_csv, pattern='*.csv')

# --- Collect all results ---
df_list = []
for csv_fn in csv_list:
    csv_id = splitext(basename(csv_fn))[0]
    tag_resn, labmod, resolution = re.search('(?<=tag)[A-Z]+', csv_id).group(0), re.search('(?<=labmod)[^_]+', csv_id).group(0), re.search('(?<=res)[0-9]+', csv_id).group(0)
    df = pd.read_csv(csv_fn, index_col=0)
    if not len(df): continue
    df.loc[:,'tagged_resn'] = tag_resn
    df.loc[:, 'labmod'] = labmod
    df.loc[:, 'resolution'] = int(resolution) * 0.01
    df_list.append(df)
raw_df = pd.concat(df_list)
raw_df = raw_df.query('nb_tags != 0').copy()  # structures without tags are not observed
raw_df.to_csv(f'{out_file_base}_raw_data.csv', index=False)

raw_df.loc[:, 'nbt_bin'] = 0
raw_df.reset_index(drop=True, inplace=True)

for (tag_resn, labmod, resolution), cdf in raw_df.groupby(['tagged_resn', 'labmod', 'resolution']):
    bin_borders = np.unique(cdf.nb_tags.quantile(np.arange(0.2, 1.0, 0.2)).to_numpy().round())
    bin_borders = np.concatenate((bin_borders, [cdf.nb_tags.max()]))
    bb_w0 = np.concatenate(([0], bin_borders))
    bin_mids = bb_w0[:-1] + (bb_w0[1:] - bb_w0[:-1]) / 2
    raw_df.loc[cdf.index, 'nbt_bin'] = cdf.nb_tags.apply(lambda x: bin_mids[np.min(np.argwhere(x <= bin_borders))])

# --- multiline plot ---
ml_df_list = []
for (nb_tags, nbt_bin, labmod, tagged_resn, mod_id, resolution), cdf in raw_df.groupby(['nb_tags', 'nbt_bin', 'labmod', 'tagged_resn', 'mod_id', 'resolution']):
    ml_df_list.append(pd.Series({
        'acc': cdf.pred.mean(), 'tagged_resn': tagged_resn, 'labmod': labmod, 'nb_tags': nb_tags, 'nbt_bin': nbt_bin, 'mod_id': mod_id,
        'resolution': resolution, 'nb_fingerprints': len(cdf), 'nb_correct': cdf.pred.sum()
    }))
ml_df = pd.concat(ml_df_list, axis=1).T
ml_df.nbt_bin = ml_df.nbt_bin.astype(float)
ml_df.acc = ml_df.acc.astype(float) * 100
ml_df.resolution = ml_df.resolution.astype(float)
mlu = ml_df.resolution.unique()
res_levels = [mlu[np.argmin(np.abs(mlu - mq))] for mq in np.quantile(ml_df.resolution, [0, 0.5, 1.0]).tolist()]
ml_df = ml_df.query(f'resolution in {res_levels}').copy()
res_lty_dict = {rl: st for rl, st in zip(res_levels, ['-', '--', 'dotted'])}

barplot_df_list = []
for (tagged_resn, labmod, resolution), cdf in raw_df.groupby(['tagged_resn', 'labmod', 'resolution']):
    if resolution not in res_levels: continue
    barplot_df_list.append(pd.Series({'acc': cdf.pred.mean() * 100, 'resolution': resolution, 'labmod': labmod}, name=(labmod, tagged_resn)))
barplot_df_list.sort(key=lambda x: sort_key[x.name], reverse=True)
barplot_df = pd.concat(barplot_df_list, axis=1).T
barplot_df.acc = barplot_df.acc.astype(float)
# barplot_df.resolution = barplot_df.resolution.astype(float)
barplot_df.to_csv(f'{out_file_base}_per_resolution_barplot_data.csv')

fig, ((perf1_ax, perf2_ax, perf3_ax),
      (subopt1_ax, subopt2_ax, subopt3_ax)) = plt.subplots(2, 3, figsize=[8.25 * 2, 2.91 * 4], sharey='all', sharex='col')
ax_dict = {('Perfect','C'): perf1_ax,
           ('Perfect','CK'): perf2_ax,
           ('Perfect','CKR'): perf3_ax,
           ('Suboptimal', 'C'): subopt1_ax,
           ('Suboptimal', 'CK'): subopt2_ax,
           ('Suboptimal', 'CKR'): subopt3_ax}


for (labmod, tagged_resn), cdf in ml_df.groupby(['labmod', 'tagged_resn']):
    sns.lineplot(x='nbt_bin', y='acc', hue='tagged_resn', style='resolution',# dashes=res_lty_dict,
                 palette=color_dict, err_style='bars', ci=68, err_kws={'capsize': 4},
                 ax=ax_dict[(labmod, tagged_resn)], data=cdf)

subopt2_ax.set_xlabel('# tags')
perf1_ax.set_ylabel('Accuracy (%)')

plt.savefig(f'{out_file_base}_per_resolution_acc_6plot.svg')
plt.close(fig=fig)

#
# for labmod, cdf in ml_df.groupby('labmod'):
#     fig, (lp_ax, bp_ax) = plt.subplots(1, 2, figsize=[8.25 * 2, 2.91 * 2], gridspec_kw={'width_ratios': [0.8,0.2]})
#     sns.lineplot(x='nbt_bin', y='acc', hue='tagged_resn', style='resolution',# dashes=res_lty_dict,
#                  palette=color_dict, err_style='bars', ci=68, err_kws={'capsize': 4},
#                  ax=lp_ax, data=cdf)
#
#     bdf = barplot_df.query(f'labmod == "{labmod}"')
#     bp_art = bp_ax.barh(np.arange(len(bdf)), bdf.acc)
#     for it, (idx, tup) in enumerate(bdf.iterrows()):
#         bp_art[it].set_edgecolor(color_dict[idx[1]])
#         bp_art[it].set_linewidth(2)
#         bp_art[it].set_linestyle(res_lty_dict[tup.resolution])
#         bp_art[it].set_facecolor('white')
#     bp_ax.set_xlabel('Overall accuracy (%)')
#
#     bp_ax.set_yticks([])
#     bp_ax.set_yticklabels([])
#
#     plt.savefig(f'{out_file_base}_per_resolution_acc_{labmod}.svg')
#     plt.close(fig=fig)



# --- lineplot ---
fig, (lp_ax, dummy_ax) = plt.subplots(1, 2, figsize=[8.25 * 2, 2.91 * 2], gridspec_kw={'width_ratios': [0.8,0.2]})
plot_df_list = []
for (resolution, labmod, tagged_resn, mod_id), cdf in raw_df.groupby(['resolution', 'labmod', 'tagged_resn', 'mod_id']):
    plot_df_list.append(pd.Series({'acc': cdf.pred.mean(), 'resolution': resolution, 'tagged_resn': tagged_resn, 'labmod': labmod,
                                   'mod_id': mod_id}))
plot_df = pd.concat(plot_df_list, axis=1).T
plot_df.resolution = plot_df.resolution.astype(float)
plot_df.acc = plot_df.acc.astype(float) * 100
plot_df.to_csv(f'{out_file_base}_lineplot_data.csv', index=False)

sns.lineplot(x='resolution', y='acc',
             hue='tagged_resn',
             style='labmod', style_order=['Perfect', 'Suboptimal'],
             palette=color_dict,
             err_style='bars', ci="sd",
             err_kws={'capsize': 4},
             data=plot_df, ax=lp_ax)

lp_ax.get_legend().remove()
lp_ax.set_ylabel('Accuracy (%)')
lp_ax.set_xlabel('$E_{FRET}$ resolution')
plt.savefig(args.out_svg, dpi=400)
