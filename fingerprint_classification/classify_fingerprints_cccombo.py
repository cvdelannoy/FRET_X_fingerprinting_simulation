import argparse, pickle
from math import inf
import pandas as pd
import numpy as np
from CorrClassifier import CorrComboClassifier


parser = argparse.ArgumentParser(description='Use a pickled rf classifier to classify fingerprints in a '
                                             'particular fold')
parser.add_argument('--leave-out-fold', type=int, required=True)
parser.add_argument('--tagged-resn', type=str, required=True)
parser.add_argument('--fp-pkl', type=str, required=True)
parser.add_argument('--rf-pkl', type=str, required=True)
parser.add_argument('--out-csv', type=str, required=True)
args = parser.parse_args()

with open(args.rf_pkl, 'rb') as fh: rf = pickle.load(fh)
with open(args.fp_pkl, 'rb') as fh: struct_dict = pickle.load(fh)
tagged_resn = list(args.tagged_resn)

max_nb_fps = max([len(struct_dict[pdb_id]['fingerprints']) for pdb_id in struct_dict])
rf_results_list = []

X_test, y_test = [], []
for pdb_id in struct_dict:
    for cvi in struct_dict[pdb_id]['fingerprints']:
        if cvi == args.leave_out_fold:
            xt = {tn: struct_dict[pdb_id]['fingerprints'][cvi][tn] for tn in tagged_resn}
            if np.all([ np.sum(xt[resn]) == 0 for resn in xt]): continue  # filter out invisible proteins
            X_test.append(xt)
            y_test.append(pdb_id)
            rf_results_list.append(pd.Series({'pdb_id': pdb_id, 'mod_id': cvi,
                                              'nb_tags': struct_dict[pdb_id]['properties']['number_of_tags']}))
if len(rf_results_list):
    rf_results_df = pd.concat(rf_results_list, axis=1).T
    pred = rf.predict(X_test)
    rf_results_df.loc[:, 'pdb_id_pred'] = pred
    rf_results_df.loc[:, 'pred'] = rf_results_df.apply(lambda x: x.pdb_id == x.pdb_id_pred, axis=1)
else:
    rf_results_df = pd.DataFrame(columns=['pdb_id', 'mod_id', 'nb_tags', 'pdb_id_pred', 'pred'])

rf_results_df.to_csv(args.out_csv)
