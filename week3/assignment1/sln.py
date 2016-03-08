import pandas as pd
from sklearn.svm import SVC


def get_X_y(df_):
    X = df_[range(1, len(df_.columns))]
    y = df_[0]
    return X, y

df = pd.read_csv('week3/assignment1/svm-data.csv', header=None)
X, y = get_X_y(df)

clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X, y)

print(clf.support_)
