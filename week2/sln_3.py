import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def get_X_y(df):
    X = df[range(1, len(df.columns))]
    y = df[0]
    return X, y


def get_score(X_train_, y_train_, X_test_, y_test_):
    clf = Perceptron(random_state=241)
    clf.fit(X_train_, y_train_)

    return accuracy_score(y_test_, clf.predict(X_test_))

df_train = pd.read_csv('week2/perceptron-train.csv', header=None)
df_test = pd.read_csv('week2/perceptron-test.csv', header=None)

X_train, y_train = get_X_y(df_train)
X_test, y_test = get_X_y(df_test)
score = get_score(X_train, y_train, X_test, y_test)
print(score)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
score_scaled = get_score(X_train_scaled, y_train, X_test_scaled, y_test)
print(score_scaled)

print(score_scaled - score)
