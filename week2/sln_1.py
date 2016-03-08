import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale

df = pd.read_csv('week2/wine.data', header=None)
X = df[range(1, len(df.columns))]
y = df[0]
kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)

scores = {}
for k in range(1, 51):
    scores[k] = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=kf).mean()

max(scores, key=scores.get)
scores[max(scores, key=scores.get)]

X = scale(X)
