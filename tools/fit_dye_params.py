import os, sys, argparse
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
import helpers as nhp

def get_cacb_from_res(res):
    ca, cb = None, None
    for atm in res.get_atoms():
        if atm.name == 'CA': ca = atm.coord
        elif atm.name == 'CB': cb = atm.coord
        if ca is not None and cb is not None:
            break
    return ca, cb

def tag_dist_fun(c, x):
    return np.linalg.norm((c[0] + c[1] * x) - (c[2] + c[3] * x))

def str2list(line, id_str, levels):
    if levels == 1:
        return np.array([float(n.strip('[]')) for n in line.replace(id_str, '').strip('[]\n').split(', ')])
    if levels == 2:
        # return [[float(nn) for nn in n.split(', ')] for n in line.replace(id_str, '').strip('[]\n').split('], [')]
        fp = [[nn for nn in n.strip('[]\n').split(', ')] for n in
              line.replace(id_str, '').strip('[]\n').split(', ')]
        fp2 = []
        for ni, n in enumerate(fp):
            sub_fp=[]
            for nn in n:
                if len(nn):
                    val = float(nn)
                else:
                    val = None
                sub_fp.append(val)
            fp2.append(sub_fp)
        return fp2


def parse_remark_list(pdb_fn, remark_id, lv):
    line_found = False
    with open(pdb_fn, 'r') as fh:
        for line in fh.readlines():
            if remark_id in line:
                line_found = True
                break
    if not line_found: return
    return str2list(line, remark_id, lv)

pdb_parser = PDBParser()

parser = argparse.ArgumentParser(description='Apply different dye distance and angle rules to existing lattice models')
parser.add_argument('--lat-in', type=str, nargs='+', required=True)
parser.add_argument('--exp-data', type=str, required=True,
                    help='Folder(s)/files containing experimental Efret values, with same names as pdbs')
parser.add_argument('--dye-dist', nargs='+', type=int, default=list(range(15,25)))
parser.add_argument('--out-dir', type=str, required=True)
args = parser.parse_args()

# Construct experimental data df
exp_df = pd.read_csv(args.exp_data, index_col=0, header=0)
exp_df.loc[:, 'distance'] = exp_df.efret.apply(nhp.get_FRET_distance)

lat_list = nhp.parse_input_dir(args.lat_in, pattern='*.pdb')
lat_list = [lfn for lfn in lat_list if 'intermediates' not in lfn]
lat_dict = {pdbid: {'lat_list': [ll for ll in lat_list if pdbid in ll], 'acc_resi': exp_df.query(f'index == "{pdbid}"').acc_resi.unique()[0]} for pdbid in exp_df.index}
out_dir = nhp.parse_output_dir(args.out_dir)

# Collect tag0 and tag coords (normalized)
y = []
x_list = []
dist_dict = {}
for pdbid in lat_dict:
    cur_lat_list, cur_acc_resi = lat_dict[pdbid]['lat_list'], lat_dict[pdbid]['acc_resi']
    if not len(cur_lat_list): continue
    dist_dict[pdbid] = []
    for cl in cur_lat_list:
        struct = pdb_parser.get_structure('pdbid', cl)
        e_fp = parse_remark_list(cl, 'REMARK   1 ENERGIES ', 1)
        e_fp_struct = []
        cur_coord_dict = {}
        for mi, mod in enumerate(struct.get_models()):
            tag_res = [res for res in mod.get_residues() if res.resname == 'TAG']
            tc_dict = {}
            tag_missing = False
            for res in tag_res:
                if res.id[1] == cur_acc_resi:
                    tag0_ca, tag0_cb = get_cacb_from_res(res)
                    if tag0_cb is None:
                        tag_missing = True
                        break
                    tag0_d = (tag0_cb - tag0_ca) / np.linalg.norm(tag0_cb - tag0_ca)
                else:
                    tag_ca, tag_cb = get_cacb_from_res(res)
                    if tag_cb is None:
                        tag_missing = True
                        break
                    tag_d = (tag_cb - tag_ca) / np.linalg.norm(tag_cb - tag_ca)
                    tc_dict[res.id[1]] = (tag_ca, tag_d)
            if tag_missing: break
            e_fp_struct.append(e_fp[mi])
            for tc in tc_dict:
                new_tup = np.array([tag0_ca, tag0_d, tc_dict[tc][0], tc_dict[tc][1]])
                if tc in cur_coord_dict:
                    cur_coord_dict[tc].append(new_tup)
                else:
                    cur_coord_dict[tc] = [new_tup]
                # cur_coord_list.append(np.array([tag0_ca, tag0_d, tc[0], tc[1]]))
        dist_dict[pdbid].append((cur_coord_dict, np.array(e_fp_struct)))


    # cur_x_dists = np.array([tag_dist_fun(crx, 15) for crx in cur_x_list])
    # qt = np.quantile(cur_x_dists, (0.4, 0.6))
    # cur_x_list = np.array(cur_x_list)[np.logical_and(cur_x_dists > qt[0], cur_x_dists < qt[1])]
    #
    # y.extend([exp_df.loc[pdbid, 'distance']] * len(cur_x_list))
    # x_list.extend(cur_x_list)

# fit
# cf = curve_fit(tag_dist_fun, x_list, y, 18)[0][0]
# with open(f'{out_dir}dye_dist_estimation.txt', 'w') as fh: fh.write(str(cf))

# plot
col_order = []
sections = np.linspace(0, 1, len(exp_df.index.unique()) + 1)
for it, pdbid in enumerate(exp_df.index.unique()):
    exp_df.loc[pdbid, 'lb'] = sections[it]
    exp_df.loc[pdbid, 'rb'] = sections[it+1]
    col_order.append(str(pdbid))
# exp_df.loc[:, 'lb'] = sections[:-1]
# exp_df.loc[:, 'rb'] = sections[1:]

for td in args.dye_dist:
    plot_df_list = []
    for pdbid in dist_dict:
        nb_tags = len(exp_df.loc[[pdbid],:])
        for tup in dist_dict[pdbid]:
            for resi in tup[0]:
                efret_mean = np.mean([nhp.get_FRET_efficiency(tag_dist_fun(x_cur, td)) for x_cur in tup[0][resi]])
                # efret_mean = np.sum(np.array([nhp.get_FRET_efficiency(tag_dist_fun(x_cur, td)) for x_cur in tup[0][resi]]) * np.tile(tup[1], nb_tags)) / (np.sum(tup[1]) * nb_tags)
                plot_df_list.append(pd.Series({'pdbid': pdbid, 'efret': efret_mean, 'resi': resi }))
    plot_df = pd.concat(plot_df_list, axis=1).T
    # sns.boxplot(y='efret', x='pdbid', color='white', data=plot_df, order=exp_df.index)
    swarm_ax = sns.stripplot(y='efret', x='pdbid', data=plot_df, order=col_order)
    for lid, tup in exp_df.iterrows():
        swarm_ax.axhline(tup.loc['efret'], tup.lb, tup.rb, color='red')

    plt.savefig(f'{out_dir}refitted_efret_hists_d{td}.svg')
    plt.close(plt.gcf())
    plot_df.to_csv(f'{out_dir}refitted_efret_hists_d{td}.csv')
