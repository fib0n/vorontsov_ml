import pandas as pd
import math
from sklearn.metrics import roc_auc_score


def get_X_y(df_):
    return df_[range(1, len(df_.columns))], df_[0]


def find_w(C=0.0):
    def get_sum(param):
        acc = 0.0
        for i in range(0, l):
            acc += y[i] * param[i] * (1.0 - (1.0 + math.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))) ** (-1))
        return k / l * acc

    w1, w2, k, l, eps, count_step = 0, 0, 0.1, len(y), 1e-10, 0
    while count_step <= 10000:
        d1 = get_sum(X[1]) - k * C * w1
        d2 = get_sum(X[2]) - k * C * w2
        if d1 ** 2 + d2 ** 2 < eps:
            break
        w1 += d1
        w2 += d2
        count_step += 1

    print(count_step)
    return w1, w2


def get_a(w1, w2):
    a = []
    for i in range(0, len(y)):
        a.append((1.0 + math.exp(-w1 * X[1][i] - w2 * X[2][i])) ** (-1))
    return a


df = pd.read_csv('week3/assignment3/data-logistic.csv', header=None)
X, y = get_X_y(df)

w1, w2 = find_w()
print(roc_auc_score(y, get_a(w1, w2)))
w1, w2 = find_w(10.0)
print(roc_auc_score(y, get_a(w1, w2)))
