from math import inf
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

class CorrClassifier(object):
    def __init__(self, comparison_data, y):
        self.data = comparison_data
        self.y = y

    def predict(self, X):
        yh = []
        for x in X:
            dists = []
            for d in self.data:
                dist = []
                for resn in d:
                    if d[resn] is None or x[resn] is None: continue
                    dist.append(np.corrcoef(x[resn], d[resn])[0, 1])
                if len(dist):
                    dists.append(np.sum(dist))
                else:
                    dists.append(-inf)
            yh.append(self.y[np.argmax(dists)])
        return yh


class KnnClassifier(object):
    def __init__(self, comparison_data, y):
        self.data = self.prep_data(comparison_data)
        self.nb_features = len(self.data[0])
        self.y = y

    def prep_data(self, X):
        out_vec = []
        for x in X:
            ov = {}
            for resn in x:
                n = x[resn]
                n[0] += 1
                ov[resn] = n
            out_vec.append(ov)
        return out_vec

    def predict(self, X):
        X = self.prep_data(X)
        yh = []
        for x in X:
            dists = []
            for d in self.data:
                dist = []
                for resn in d:
                    dd = 1 - np.corrcoef(x[resn], d[resn])[0, 1]
                    if np.isnan(dd): dd = 1
                    dist.append(dd)
                # dists.append(dist)
                dists.append(np.sqrt(np.sum(np.array(dist) ** 2)))
            # knn = KNeighborsClassifier(n_neighbors=3).fit(np.array(dists), self.y)
            # yh.append(knn.predict(np.zeros(self.nb_features).reshape(1,-1)))
            yh.append(self.y[np.argmin(dists)])
        return yh


class CorrTreeClassifier(object):
    def __init__(self, comparison_data, y):
        self.data = comparison_data
        self.encoder = LabelEncoder().fit(y)
        self.y = self.encoder.transform(y)
        self.xgb = XGBClassifier(use_label_encoder=False, objective='multi:softmax', eval_metric='mlogloss')
        self.train()

    def train(self):
        features = self.prep_data(self.data)
        self.xgb.fit(features, self.y)

    def prep_data(self, X):
        feature_vectors = []
        for x in X:
            dists = []
            for d in self.data:
                dist = []
                for resn in d:
                    if d[resn] is None or x[resn] is None: continue
                    dist.append(np.corrcoef(x[resn], d[resn])[0, 1])
                if len(dist):
                    dists.append(np.sum(dist))
                else:
                    dists.append(0)
            feature_vectors.append(dists)
        return np.vstack(feature_vectors)

    def predict(self, X):
        features = self.prep_data(X)
        pred = self.xgb.predict(features)
        return self.encoder.inverse_transform(pred)


class CorrComboClassifier(object):
    def __init__(self, comparison_data, y, tagged_resn):
        self.tagged_resn = list(tagged_resn)  # stores order of residues, to be sure
        self.data = comparison_data
        self.encoder = LabelEncoder().fit(y)
        self.y = self.encoder.transform(y)
        self.xgb = XGBClassifier(use_label_encoder=False, objective='multi:softmax', eval_metric='mlogloss')
        self.train()

    def train(self):
        features = self.prep_data(self.data)
        self.xgb.fit(features, self.y)

    def prep_data(self, X):
        feature_vectors = []
        for x in X:
            dists = []
            efret = [x[resn] for resn in self.tagged_resn]
            for d in self.data:
                dist = []
                for resn in d:
                    if np.sum(x[resn]) == 0 or np.sum(d[resn]) == 0: continue
                    dist.append(np.corrcoef(x[resn], d[resn])[0, 1])
                if len(dist):
                    dists.append(np.sum(dist))
                else:
                    dists.append(0)
            feature_vectors.append(np.concatenate([dists] + efret))
        return np.vstack(feature_vectors)

    def predict(self, X):
        features = self.prep_data(X)
        pred = self.xgb.predict(features)
        return self.encoder.inverse_transform(pred)
