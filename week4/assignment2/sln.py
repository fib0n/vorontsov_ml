import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv('week4/assignment2/close_prices.csv')
X = df[range(1, len(df.columns))]
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)

variance = 0.0
count = 0
for v in pca.explained_variance_ratio_:
    if variance >= 0.9:
        break
    else:
        variance += v
        count += 1

print(count)
X_new = pca.transform(X)
first_component = X_new[::, 0]

df_djia = pd.read_csv('week4/assignment2/djia_index.csv')
index = df_djia['^DJI']

print(np.corrcoef(first_component, index))

pca.components_[0].min()
pca.components_[0].max()

pca.components_[0].argmax()
