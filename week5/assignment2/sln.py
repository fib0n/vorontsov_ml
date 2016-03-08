import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


def get_X_y(df_):
    X_ = df_.ix[:, 1:]
    y_ = df_['Activity']
    return X_.values, y_.values


data = pd.read_csv('week5/assignment2/gbm-data.csv')
X, y = get_X_y(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


def get_sigmoid(pred):
    return 1.0 / (1.0 + np.exp(-pred))


def get_loss(X_, y_, clf_):
    loss = []
    for y_pred in clf_.staged_decision_function(X_):
        loss.append(log_loss(y_, get_sigmoid(y_pred)))
    return loss


for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=250, random_state=241, verbose=False)
    clf.fit(X_train, y_train)

    test_loss = get_loss(X_test, y_test, clf)
    train_loss = get_loss(X_train, y_train, clf)

    print(learning_rate)
    print(np.argmin(test_loss) + 1)
    print(min(test_loss))
    print(log_loss(y_test, get_sigmoid(clf.predict_proba(X_test))))

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

rclf = RandomForestClassifier(n_estimators=37, random_state=241)
rclf.fit(X_train, y_train)
print(log_loss(y_test, rclf.predict_proba(X_test)))
