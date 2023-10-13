import argparse, os, sys, re, pickle

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from helpers import parse_input_dir


def dir2df(dir_fn, regex):
    df_list = []
    data_dict = {re.search(regex, fn).group(0): np.loadtxt(fn) for fn in parse_input_dir(dir_fn, pattern='*.txt')}

    for fn in parse_input_dir(dir_fn, pattern='*.txt'):
        efret_array = np.loadtxt(fn)
        if efret_array.ndim == 1: efret_array = efret_array.reshape(-1,1)
        nb_peaks = efret_array.shape[1]
        df_dict = {f'efret{i}': efret_array[:, i] for i in range(nb_peaks)}
        df_dict['cl_id'] = re.search(regex, fn).group(0)
        df_list.append(pd.DataFrame(df_dict))
    return pd.concat(df_list)


def main(train_dir, test_dir, pred_csv, regex, plot=False):
    train_df = dir2df(train_dir, regex)
    test_df = dir2df(test_dir, regex)
    feature_names = train_df.columns[:-1]

    # SVM
    mod = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    mod.fit(train_df.loc[:, feature_names], train_df.cl_id)
    test_df.loc[:, 'pred'] = mod.predict(test_df.loc[:, feature_names])

    with open(f'{Path(pred_csv).parent}/{Path(pred_csv).stem}_model.pkl', 'wb') as fh:
        pickle.dump(mod, fh)
    test_df.to_csv(pred_csv, index=False)
    test_summary_df = pd.DataFrame({'counts': test_df.groupby('pred').count().cl_id})
    test_summary_df.loc[:, 'frac'] = test_summary_df.counts / test_summary_df.counts.sum()
    test_summary_df.to_csv(os.path.splitext(pred_csv)[0] + '_summary.csv')

    nb_mols = len(test_df)
    bs_list = []
    for bi in range(1000):
        cdf = test_df.iloc[np.random.randint(nb_mols, size=nb_mols)].reset_index(drop=True)
        summary_cdf = pd.DataFrame({'counts': cdf.groupby('pred').count().cl_id})
        summary_cdf.loc[:, 'frac'] = summary_cdf.counts / summary_cdf.counts.sum()
        bs_list.append(summary_cdf)
    test_bs_df = pd.concat(bs_list).reset_index()
    test_bs_df.to_csv(f'{Path(pred_csv).parent}/{Path(pred_csv).stem}_bootstrapped_fractions.csv')
    test_bs_df.loc[:, 'dummy'] = 'A'

    if plot:
        plt.subplots(figsize=(10,5))
        sns.histplot(x='efret0', hue='pred', bins=np.arange(0,1,0.01), data=test_df)
        plt.savefig(f'{Path(pred_csv).parent}/{Path(pred_csv).stem}_hist.svg')
        plt.close()

        for pid, cdf in test_df.groupby('pred'):
            np.savetxt(f'{os.path.splitext(pred_csv)[0]}_hist_{pid}.txt', cdf.efret0)

        fig, ax = plt.subplots(figsize=(5,5))
        sns.barplot(x='dummy', hue='pred', y='frac', errorbar=('se',2), data=test_bs_df)
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_ylabel('Fraction')
        plt.savefig(f'{Path(pred_csv).parent}/{Path(pred_csv).stem}_counts.svg')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier to detect asyn classes')
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--pred-csv', type=str, required=True)
    parser.add_argument('--regex', type=str, default='[A-Z][0-9]+[A-Z](?=.txt)')
    args = parser.parse_args()

    main(args.train_dir, args.test_dir, args.pred_csv, args.regex, True)
