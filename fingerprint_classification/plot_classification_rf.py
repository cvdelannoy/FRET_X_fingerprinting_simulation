import os, sys, argparse, re
import pandas as pd
import numpy as np
from os.path import basename, splitext, dirname
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles
from itertools import chain

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])

from helpers import parse_input_dir, aa_dict

mpl.rc('font', **{'size': 16})

color_dict = {'C':'#66c2a5',
              'CK': '#fc8d62',
              'KR': '#e78ac3',
              'CKR': '#8da0cb'}

sort_key = {('Perfect', 'CKR'): 0,
            ('Perfect', 'CK'): 2,
            ('Perfect', 'C'): 4,
            ('Suboptimal', 'CKR'): 1,
            ('Suboptimal', 'CK'): 3,
            ('Suboptimal', 'C'): 5}

venn_order = [['C'], ['CK'], ['C', 'CK'], ['CKR'], ['C', 'CKR'], ['CK', 'CKR'], ['C', 'CK', 'CKR']]
venn_labels = ['C', 'CK', 'CKR']
venn_colors = {'100': color_dict['C'], '010': color_dict['CK'], '001': color_dict['CKR']}

bp_order = ['C', 'CK', 'CKR']

parser = argparse.ArgumentParser(description='Plot results rf classification results, figure 1 part A')
parser.add_argument('--result-csv', type=str, nargs='+', required=True)
parser.add_argument('--define-good', type=float, default=0.5,
                    help='Define what fraction of fingerprints must be correctly predicted before it is marked '
                         'as "well-predicted"')
parser.add_argument('--out-svg', type=str, required=True)
args = parser.parse_args()

out_file_base = splitext(args.out_svg)[0]
csv_list = parse_input_dir(args.result_csv, pattern='*.csv')

# --- Collect all results ---
df_list = []
for csv_fn in csv_list:
    csv_id = splitext(basename(csv_fn))[0]
    tag_resn, labmod = re.search('(?<=tag)[A-Z]+', csv_id).group(0), re.search('(?<=labmod)[^_]+', csv_id).group(0)
    df = pd.read_csv(csv_fn, index_col=0)
    if not len(df): continue
    df.loc[:,'tagged_resn'] = tag_resn
    df.loc[:, 'labmod'] = labmod
    df_list.append(df)
raw_df = pd.concat(df_list)
raw_df = raw_df.query('nb_tags != 0').copy()  # structures without tags are not observed
raw_df.loc[:, 'nbt_bin'] = 0
raw_df.reset_index(drop=True, inplace=True)

for (tag_resn, labmod), cdf in raw_df.groupby(['tagged_resn', 'labmod']):
    bin_borders = np.unique(cdf.nb_tags.quantile(np.arange(0.2, 1.0, 0.2)).to_numpy().round())
    bin_borders = np.concatenate((bin_borders, [cdf.nb_tags.max()]))
    bb_w0 = np.concatenate(([0], bin_borders))
    bin_mids = bb_w0[:-1] + (bb_w0[1:] - bb_w0[:-1]) / 2
    raw_df.loc[cdf.index, 'nbt_bin'] = cdf.nb_tags.apply(lambda x: bin_mids[np.min(np.argwhere(x <= bin_borders))])

# --- figures on well-predicted structures ---

# --- Venn diagram ---
tagged_resn_levels = raw_df.tagged_resn.unique()
labmod_levels = raw_df.labmod.unique()
venn_df = pd.DataFrame(False, index=raw_df.pdb_id.unique(),
                       columns=pd.MultiIndex.from_product([labmod_levels, tagged_resn_levels], names=['labmod', 'tagged_resn']))
for (pdb_id, labmod, tag_resn), cdf in raw_df.groupby(['pdb_id', 'labmod', 'tagged_resn']):
    # venn_df.loc[pdb_id,( labmod, tag_resn)] = cdf.pred.all()
    venn_df.loc[pdb_id, (labmod, tag_resn)] = cdf.pred.mean() > args.define_good
for cc in venn_order:
    c_name = '_'.join(cc)
    include_array = np.logical_and.reduce([venn_df.loc[:, 'Perfect'].loc[:, c] for c in cc])
    exclude_labels = [vl for vl in venn_labels if vl not in cc]
    if len(exclude_labels):
        exclude_array = np.logical_or.reduce([venn_df.loc[:, 'Perfect'].loc[:, c] for c in exclude_labels])
        include_array = np.logical_and(include_array, np.invert(exclude_array))

    venn_df.loc[:, ('Perfect_venn', c_name)] = include_array

# save ID lists of well-predicted proteins
labmodPerfect_wpids = venn_df.index[np.any([venn_df[('Perfect', resn)] for resn in venn_df.loc[:, 'Perfect'].columns], axis=0)]
np.savetxt(f'{out_file_base}_wellpredicted_labmodPerfect.txt', labmodPerfect_wpids, fmt="%s")
if ' Suboptimal' in venn_df:
    labmodSuboptimal_wpids = venn_df.index[np.all([venn_df[('Suboptimal', resn)] for resn in venn_df.loc[:, 'Perfect'].columns], axis=0)]
    np.savetxt(f'{out_file_base}_wellpredicted_labmodSuboptimal.txt', labmodSuboptimal_wpids, fmt="%s")

venn_df.loc['total', :] = venn_df.sum(axis=0)
venn_df.to_csv(f'{out_file_base}_wellpredicted.csv')
venn_array = [int(venn_df.loc['total', 'Perfect_venn'].get('_'.join(tr), default=0)) for tr in venn_order]

# construct accuracy barplot df
barplot_df_list = []
for (tagged_resn, labmod), cdf in raw_df.groupby(['tagged_resn', 'labmod']):
    barplot_df_list.append(pd.Series({'acc': cdf.pred.mean() * 100}, name=(labmod, tagged_resn)))
barplot_df_list.sort(key=lambda x: sort_key[x.name], reverse=True)
barplot_df = pd.concat(barplot_df_list, axis=1).T
barplot_df.to_csv(f'{out_file_base}_barplot_data.csv')

# --- discernible structures plot ---
fig_wp, (ax_wpbar, ax_venn) = plt.subplots(1,2, figsize=[8.25 * 2, 2.91 * 2], gridspec_kw={'width_ratios':[0.6, 0.4]})

v = venn3(venn_array, set_labels=venn_labels, alpha=1, ax=ax_venn)
venn3_circles(venn_array, linewidth=2, color='black', ax=ax_venn)
for k in venn_colors:
    patch = v.get_patch_by_id(k)
    if patch is not None: patch.set_color(venn_colors[k])

# wpbar_list = []
# if ' Suboptimal' in venn_df:
#     wpbar_df = pd.DataFrame({
#         'nb_prots': list(chain.from_iterable([(venn_df.loc['total', 'Perfect'].loc[resn], venn_df.loc['total', 'Suboptimal'].loc[resn])
#                                                for resn in bp_order[::-1]])),
#         'labmod': ['Perfect', 'Suboptimal'] * len(bp_order),
#         'tagged_resn': list(chain.from_iterable([[resn] * 2 for resn in bp_order[::-1]]))
#     })
# else:
#     wpbar_df = pd.DataFrame({
#         'nb_prots': list(chain.from_iterable(
#             [(venn_df.loc['total', 'Perfect'].loc[resn], 0)
#              for resn in bp_order[::-1]])),
#         'labmod': ['Perfect', 'Suboptimal'] * len(bp_order),
#         'tagged_resn': list(chain.from_iterable([[resn] * 2 for resn in bp_order[::-1]]))
#     })
# wpbar_df.set_index(['labmod', 'tagged_resn'], inplace=True)
#
#
# bp_art = ax_wpbar.bar(np.arange(len(wpbar_df)), wpbar_df.nb_prots)
# for it, ((labmod, tr), tup) in enumerate(wpbar_df.iterrows()):
#     bp_art[it].set_edgecolor(color_dict[tr])
#     bp_art[it].set_linewidth(2)
#     if labmod == 'Perfect':
#         bp_art[it].set_linestyle('-')
#         bp_art[it].set_facecolor(color_dict[tr])
#     else:
#         bp_art[it].set_linestyle('--')
#         bp_art[it].set_facecolor('white')
#
# ax_wpbar.axhline(len(raw_df.pdb_id.unique()), color='black', ls='--', lw=1)
# ax_wpbar.set_ylabel('# Discernible proteins')
# ax_wpbar.set_xlabel(None)
# ax_wpbar.set_xticks([])
# ax_wpbar.set_xticklabels([])
#
# fig_wp.savefig(f'{out_file_base}_wellpredicted_plots.svg')

bp_art = ax_wpbar.bar(np.arange(len(barplot_df)),barplot_df.acc)
for it, (idx, tup) in enumerate(barplot_df.iterrows()):
    bp_art[it].set_edgecolor(color_dict[idx[1]])
    bp_art[it].set_linewidth(2)
    if idx[0] == 'Perfect':
        bp_art[it].set_linestyle('-')
        bp_art[it].set_facecolor(color_dict[idx[1]])
    else:
        bp_art[it].set_linestyle('--')
        bp_art[it].set_facecolor('white')
ax_wpbar.set_ylabel('Identification accuracy (%)')
ax_wpbar.set_xlabel(None)
ax_wpbar.set_xticks([])
ax_wpbar.set_xticklabels([])
plt.savefig(args.out_svg, dpi=400)
plt.tight_layout()
plt.close(fig=fig_wp)

# --- lineplot data ---
plot_df_list = []
for (nb_tags, nbt_bin, labmod, tagged_resn, mod_id), cdf in raw_df.groupby(['nb_tags', 'nbt_bin', 'labmod', 'tagged_resn', 'mod_id']):
    plot_df_list.append(pd.Series({
        'acc': cdf.pred.mean(), 'tagged_resn': tagged_resn, 'labmod': labmod, 'nb_tags': nb_tags, 'nbt_bin': nbt_bin, 'mod_id': mod_id,
        'nb_fingerprints': len(cdf), 'nb_correct': cdf.pred.sum()
    }))
plot_df = pd.concat(plot_df_list, axis=1).T
plot_df.acc = plot_df.acc.astype(float) * 100
plot_df.nb_tags = plot_df.nb_tags.astype(int)
plot_df.to_csv(f'{out_file_base}_lineplot_data.csv', index=False)

# --- accuracy plots ---
fig, lp_ax = plt.subplots(1, 1, figsize=[8.25 * 2, 2.91 * 2])
# fig, (lp_ax, bp_ax) = plt.subplots(1, 2, figsize=[8.25 * 2, 2.91 * 2], gridspec_kw={'width_ratios': [0.8,0.2]})

sns.lineplot(x='nbt_bin', y='acc', hue='tagged_resn', style='labmod', style_order=['Perfect', 'Suboptimal'],
             palette=color_dict, markers=['o', 'o'], err_style='bars', ci=68, err_kws={'capsize': 4},
             ax=lp_ax, data=plot_df)

lp_ax.set_ylim([0, 100])
lp_ax.get_legend().remove()
lp_ax.set_ylabel('Accuracy (%)')
lp_ax.set_xlabel('# tags')

# # --- barplot ---
#
# bp_art = bp_ax.barh(np.arange(len(barplot_df)),barplot_df.acc)
# for it, (idx, tup) in enumerate(barplot_df.iterrows()):
#     bp_art[it].set_edgecolor(color_dict[idx[1]])
#     bp_art[it].set_linewidth(2)
#     if idx[0] == 'Perfect':
#         bp_art[it].set_linestyle('-')
#         bp_art[it].set_facecolor(color_dict[idx[1]])
#     else:
#         bp_art[it].set_linestyle('--')
#         bp_art[it].set_facecolor('white')
# bp_ax.set_xlabel('Overall accuracy (%)')
#
# bp_ax.set_yticks([])
# bp_ax.set_yticklabels([])
# plt.savefig(args.out_svg, dpi=400)
