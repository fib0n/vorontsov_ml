import pandas as pd
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor


def get_X_y(df):
    X = df[range(0, len(df.columns) - 1)]
    y = df[[len(df.columns) - 1]]
    return X, y


df = pd.read_csv('week5/assignment1/abalone.csv')
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X, y = get_X_y(df)

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=1)
for tree in range(1, 51):
    clf = RandomForestRegressor(n_estimators=tree, random_state=1)
    # clf.fit(X, y)
    # predictions = clf.predict(X)
    score = cross_val_score(clf, X, y['Rings'], cv=kf, scoring='r2').mean()
    if score > 0.52:
        print(tree)
        print(score)
        break
