import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

df = pd.read_csv('week3/assignment4/classification.csv')

TP = len(df[(df['true'] == 1) & (df['pred'] == 1)])
FP = len(df[(df['true'] == 0) & (df['pred'] == 1)])
FN = len(df[(df['true'] == 1) & (df['pred'] == 0)])
TN = len(df[(df['true'] == 0) & (df['pred'] == 0)])

print(accuracy_score(df['true'], df['pred']))
print(precision_score(df['true'], df['pred']))
print(recall_score(df['true'], df['pred']))
print(f1_score(df['true'], df['pred']))


df = pd.read_csv('week3/assignment4/scores.csv')

for col in ['score_logreg', 'score_svm', 'score_knn', 'score_tree']:
    print(roc_auc_score(df['true'], df[col]))

for col in ['score_logreg', 'score_svm', 'score_knn', 'score_tree']:
    precision, recall, thresholds = precision_recall_curve(df['true'], df[col])
    max_precision = -1
    for i in range(0, len(thresholds)):
        if recall[i] > 0.7 and precision[i] > max_precision:
            max_precision = precision[i]
    print(col)
    print(max_precision)
