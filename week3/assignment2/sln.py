import numpy as np
from sklearn import datasets, svm, grid_search
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
y = newsgroups.target

v = TfidfVectorizer()
X_train = v.fit_transform(X)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = svm.SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=5)
gs.fit(X_train, y)

clf = svm.SVC(kernel='linear', C=gs.best_params_['C'], random_state=241)
clf.fit(X_train, y)

n = 10
best_n_coef = np.argsort(np.absolute(np.asarray(clf.coef_.todense()).reshape(-1)))[::-1][:n]
words = [v.get_feature_names()[i] for i in best_n_coef]
words.sort()
print(words)
