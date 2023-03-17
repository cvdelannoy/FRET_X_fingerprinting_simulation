import argparse, re, sys, os

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
    confusion_dict = {vn:{vn: 0.0 for vn in var_names} for vn in var_names}
    confusion_dict_sd = deepcopy(confusion_dict)
    for (true_id, pred_id), sdf in df.groupby([true_var, pred_var]):
        fold_sizes = {nb_fold: len(ssdf) for nb_fold, ssdf in df.query(f'{true_var} == "{true_id}"').groupby(fold_var)}
        frac_pred = np.array([len(ssdf) / fold_sizes[nb_fold] for nb_fold, ssdf in sdf.groupby(fold_var)])
        confusion_dict[true_id][pred_id] = np.mean(frac_pred) if np.any(frac_pred > 0) else 0
        confusion_dict_sd[true_id][pred_id] = np.std(frac_pred)
    heat_df = pd.DataFrame.from_dict(confusion_dict).T
    heat_df_sd = pd.DataFrame.from_dict(confusion_dict_sd).T
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

def plot_precision_recall(df, true_var, pred_var, out_dir):
    recall = {cl_id: len(sdf.query(f'{true_var} == {pred_var}')) / len(sdf) for cl_id, sdf in df.groupby(true_var)}
    precision = {cl_id: len(sdf.query(f'{true_var} == {pred_var}')) / len(sdf) for cl_id, sdf in df.groupby(pred_var)}
    pr_df = pd.DataFrame.from_dict({'precision': precision, 'recall': recall})
    pr_df.to_csv(f'{out_dir}precision_recall.csv')
    fig, ax = plt.subplots(figsize=(5,5))
    sns.scatterplot(x='recall', y='precision', data=pr_df, ax=ax)
    for lab, tup in pr_df.iterrows():
        ax.text(tup.recall + 0.01, tup.precision, str(lab))
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    fig.savefig(f'{out_dir}precision_recall.svg')


def get_stats(df, true_var, pred_var, out_dir):
    overall_acc = np.mean(df.loc[:, true_var] == df.loc[:, pred_var])
    with open(f'{out_dir}stats.yaml', 'w') as fh:
        fh.write(f'''
overall_acc: {overall_acc}
        ''')


def sample_dict(in_dict, nb_iters):
    return {cl_id: [in_dict[cl_id][np.random.randint(len(in_dict[cl_id]), size=len(in_dict[cl_id]))] for _ in range(nb_iters)]
            for cl_id in in_dict}


def main(train_dict, test_dict, nb_folds, out_dir, regex, cores, sort_asyn, order):

    train_dict_bs = sample_dict(train_dict, nb_folds)
    test_dict_bs = sample_dict(test_dict, nb_folds)

    # make directories, split data
    folds_dir = parse_output_dir(out_dir + 'folds')
    for fi in range(nb_folds):
        cur_out_dir = parse_output_dir(folds_dir + str(fi))
        tst_dir = parse_output_dir(cur_out_dir + 'test')
        train_dir = parse_output_dir(cur_out_dir + 'train')
        for cl_id in train_dict:
            np.savetxt(f'{tst_dir}{cl_id}.txt', test_dict_bs[cl_id][fi])
            np.savetxt(f'{train_dir}{cl_id}.txt',train_dict_bs[cl_id][fi])

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
    if sort_asyn:
        var_names = sorted(pred_df.cl_id.unique(), key=lambda x: int(re.search('(?<=[A-Z])[0-9]+(?=[A-Z])', x).group(0)))
    elif order:
        var_names = order
    else:
        var_names = sorted(pred_df.cl_id.unique())
    get_stats(pred_df, 'cl_id', 'pred', out_dir)
    plot_confusion_heatmap(pred_df, 'cl_id', 'pred', 'fold', var_names, out_dir)
    plot_precision_recall(pred_df, 'cl_id', 'pred', out_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train & evaluate aSyn classification.')
    parser.add_argument('--train', type=str, nargs='+', required=True)
    parser.add_argument('--test', type=str, nargs='+', required=True)
    parser.add_argument('--regex', type=str, default='[A-Z][0-9]+[A-Z](?=.txt)')
    parser.add_argument('--nb-folds',type=int, default=10)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--nb-cores', type=int, default=4)
    parser.add_argument('--sort-asyn', action='store_true')
    parser.add_argument('--order', type=str, nargs='+')
    args = parser.parse_args()
    out_dir = parse_output_dir(args.out_dir)
    train_dict = {re.search(args.regex, fn).group(0): np.loadtxt(fn) for fn in parse_input_dir(args.train, pattern='*.txt')}
    test_dict = {re.search(args.regex, fn).group(0): np.loadtxt(fn) for fn in parse_input_dir(args.test, pattern='*.txt')}
    main(train_dict, test_dict, args.nb_folds, out_dir, args.regex, args.nb_cores, args.sort_asyn, args.order)

