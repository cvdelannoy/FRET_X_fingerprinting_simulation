import argparse, os, sys, re

from pathlib import Path
import numpy as np
import pandas as pd
import pomegranate as pg
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
from helpers import parse_input_dir


def dir2df(dir_fn, regex):
    df_list = []
    data_dict = {re.search(regex, fn).group(0): np.loadtxt(fn) for fn in parse_input_dir(dir_fn, pattern='*.txt')}

    for fn in parse_input_dir(dir_fn, pattern='*.txt'):
        df_list.append(pd.DataFrame({'efret': np.loadtxt(fn), 'cl_id': re.search(regex, fn).group(0)}))
    return pd.concat(df_list)

def main(train_dir, test_dir, pred_csv, regex):
    train_df = dir2df(train_dir, regex)
    test_df = dir2df(test_dir, regex)

    # # balance dataset
    # nb_ex = min([len(sdf) for _, sdf in train_df.groupby('cl_id')])
    # train_df = pd.concat([sdf.sample(nb_ex) for _, sdf in train_df.groupby('cl_id')]).reset_index(drop=True)

    # # RFC
    # mod = RandomForestClassifier()
    # mod.fit(train_df.efret.to_numpy().reshape(-1, 1), train_df.cl_id)
    # test_df.loc[:, 'pred'] = mod.predict(test_df.efret.to_numpy().reshape(-1, 1))

    # SVM
    mod = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    mod.fit(train_df.efret.to_numpy().reshape(-1,1), train_df.cl_id)
    test_df.loc[:, 'pred'] = mod.predict(test_df.efret.to_numpy().reshape(-1,1))

    # # Naive bayes
    # class_list = list(set(train_df.cl_id))
    # dist_dict = {cli: pg.distributions.NormalDistribution.from_samples(train_df.query(f'cl_id == "{cli}"').efret) for cli in class_list}
    # for dn in dist_dict: dist_dict[dn].name = dn
    # mod = pg.NaiveBayes(list(dist_dict.values()))
    #
    # test_df.loc[:, 'pred'] = [class_list[pi] for pi in mod.predict(test_df.efret.to_numpy().reshape(-1,1))]
    # mod_fn = f'{str(Path(pred_csv).parent)}/{Path(pred_csv).stem}_mod.json'
    # with open(mod_fn, 'w') as fh: fh.write(mod.to_json())

    test_df.to_csv(pred_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier to detect asyn classes')
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--pred-csv', type=str, required=True)
    parser.add_argument('--regex', type=str, default='[A-Z][0-9]+[A-Z](?=.txt)')
    args = parser.parse_args()

    main(args.train_dir, args.test_dir, args.pred_csv, args.regex)
