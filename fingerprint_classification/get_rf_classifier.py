import os, sys, argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])

parser = argparse.ArgumentParser(description='construct a random forest classifier, leave out a designated fold')
parser.add_argument('--leave-out-fold', type=int, required=True)
parser.add_argument('--fp-pkl', type=str, required=True)
parser.add_argument('--tagged-resn', type=str, required=True)
parser.add_argument('--out-pkl', type=str, required=True)

args = parser.parse_args()
with open(args.fp_pkl, 'rb') as fh: struct_dict = pickle.load(fh)
lov = args.leave_out_fold
tagged_resn = list(args.tagged_resn)

X_train, y_train = [], []
for pdb_id in struct_dict:
    # xt = [x for i, x in enumerate(struct_dict[pdb_id]['fingerprints'].values()) if i != args.leave_out_fold]
    xt = [np.concatenate([struct_dict[pdb_id]['fingerprints'][i][tn] for tn in tagged_resn])
           for i in struct_dict[pdb_id]['fingerprints'] if i != args.leave_out_fold]
    xt = [xc for xc in xt if np.sum(xc) > 0]
    yt = [pdb_id] * len(xt)
    X_train.extend(xt)
    y_train.extend(yt)
rf = RandomForestClassifier(n_estimators=1000, max_depth=2).fit(X_train, y_train)

with open(args.out_pkl, 'wb') as fh:
    pickle.dump(rf, fh)
