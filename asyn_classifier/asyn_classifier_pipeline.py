import argparse, re, sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from jinja2 import Template
from pathlib import Path
from snakemake import snakemake
from copy import deepcopy

sys.path.append(str(Path(__file__).resolve().parents[1]))
from helpers import parse_input_dir, parse_output_dir

__dirname__ = str(Path(__file__).resolve().parent)

def numeric_timestamp():
    return int(datetime.now().strftime('%H%M%S%f'))

def plot_confusion_heatmap(df, true_var, pred_var, fold_var, var_names, out_dir):
    confusion_dict = {vn:{vn:[] for vn in var_names} for vn in var_names}
    confusion_dict_sd = deepcopy(confusion_dict)
    for (true_id, pred_id), sdf in df.groupby([true_var, pred_var]):
        fold_sizes = {nb_fold: len(ssdf) for nb_fold, ssdf in df.query(f'{true_var} == "{true_id}"').groupby(fold_var)}
        frac_pred = [len(ssdf) / fold_sizes[nb_fold] for nb_fold, ssdf in sdf.groupby(fold_var)]
        confusion_dict[true_id][pred_id] = np.mean(frac_pred)
        confusion_dict_sd[true_id][pred_id] = np.std(frac_pred)
    heat_df = pd.DataFrame.from_dict(confusion_dict)
    heat_df_sd = pd.DataFrame.from_dict(confusion_dict_sd)
    annot_mat = (heat_df.round(2).astype(str) + '\n±' + heat_df_sd.round(2).astype(str))
    heat_df.rename_axis('Truth', axis='rows', inplace=True)
    heat_df.rename_axis('Predicted', axis='columns', inplace=True)
    heat_df.to_csv(f'{out_dir}heat_df.csv')
    annot_mat.to_csv(f'{out_dir}heat_df_annot.csv')
    for cmap in ('Blues', 'Oranges', 'Greens'):
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(data=heat_df, annot=annot_mat, fmt='s', cmap=cmap)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(title='RDS')
        plt.savefig(f'{out_dir}heatmap_confmat_{cmap}.svg', dpi=400)
        plt.close(fig)



def main(data_dict, nb_folds, out_dir, regex, cores):

    for cl_id in data_dict:
        np.random.shuffle(data_dict[cl_id])
        data_dict[cl_id] = np.array_split(data_dict[cl_id], nb_folds)

    # make directories, split data
    folds_dir = parse_output_dir(out_dir + 'folds')
    for fi in range(nb_folds):
        cur_out_dir = parse_output_dir(folds_dir + str(fi))
        tst_dir = parse_output_dir(cur_out_dir + 'test')
        train_dir = parse_output_dir(cur_out_dir + 'train')
        for cl_id in data_dict:
            cur_data = deepcopy(data_dict[cl_id])
            np.savetxt(f'{tst_dir}{cl_id}.txt', cur_data.pop(fi))
            np.savetxt(f'{train_dir}{cl_id}.txt', np.concatenate(cur_data))

    with open(f'{__dirname__}/asyn_classifier_pipeline.sf', 'r') as fh: sm_template = fh.read()
    sm_txt = Template(sm_template).render(
        __scriptdir__=__dirname__,
        work_dir=out_dir,
        folds_dir=folds_dir,
        nb_folds=nb_folds,
        regex=regex
    )
    sm_fn = f'{out_dir}asyn_classifier_pipeline{numeric_timestamp()}.sf'
    with open(sm_fn, 'w') as fh: fh.write(sm_txt)
    snakemake(sm_fn, cores=cores)
    pred_df = pd.read_csv(f'{out_dir}asyn_test_predicted_all.csv', index_col=0)
    var_names = sorted(pred_df.cl_id.unique(), key=lambda x: int(re.search('(?<=[A-Z])[0-9]+(?=[A-Z])', x).group(0)))
    plot_confusion_heatmap(pred_df, 'cl_id', 'pred', 'fold', var_names, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train & evaluate aSyn classification.')
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--regex', type=str, default='[A-Z][0-9]+[A-Z](?=.txt)')
    parser.add_argument('--nb-folds',type=int, default=10)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--nb-cores', type=int, default=4)
    args = parser.parse_args()
    out_dir = parse_output_dir(args.out_dir)
    fn_list = parse_input_dir(args.in_dir, pattern='*.txt')
    data_dict = {re.search(args.regex, fn).group(0): np.loadtxt(fn) for fn in fn_list}
    main(data_dict,args.nb_folds, out_dir, args.regex, args.nb_cores)

