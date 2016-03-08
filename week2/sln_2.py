import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

boston = datasets.load_boston()
X = scale(boston.data)
y = boston.target
kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)

scores = {}
for p in np.linspace(1, 10, 200):
    scores[p] = cross_val_score(KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, ), X, y, cv=kf, scoring='mean_squared_error').mean()

max(scores, key=scores.get)
scores[max(scores, key=scores.get)]
