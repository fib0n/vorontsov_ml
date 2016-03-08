import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

df = pd.read_csv('week4/assignment1/salary-train.csv')
X = df[['FullDescription', 'LocationNormalized', 'ContractTime']]
y_train = df['SalaryNormalized']

X['FullDescription'] = X['FullDescription'].str.lower().replace('[^a-z0-9]', ' ', regex=True)

tfidf_vectorizer = TfidfVectorizer(min_df=5)
X_train_description = tfidf_vectorizer.fit_transform(X['FullDescription'])

X['LocationNormalized'].fillna('nan', inplace=True)
X['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_train_categ = enc.fit_transform(X[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_train = hstack([X_train_description, X_train_categ]).toarray()

clf = Ridge(alpha=1.0, solver='lsqr')
clf.fit(X_train, y_train)

df_test = pd.read_csv('salary-test-mini.csv')
X_test = df_test[['FullDescription', 'LocationNormalized', 'ContractTime']]
X_test['FullDescription'] = X_test['FullDescription'].str.lower().replace('[^a-z0-9]', ' ', regex=True)
X_test_description = tfidf_vectorizer.transform(X_test['FullDescription'])
X_test['LocationNormalized'].fillna('nan', inplace=True)
X_test['ContractTime'].fillna('nan', inplace=True)
X_test_categ = enc.transform(X_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_description, X_test_categ]).toarray()
print(clf.predict(X_test))
