import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('week1/titanic.csv', header=None)
data_filtered = data[False == np.isnan(data['Age'])]
x = pd.DataFrame(data_filtered, columns=['Pclass', 'Fare', 'Age', 'Sex'])
mapping = {'male': 0, 'female': 1}
x = x.replace({'Sex': mapping})
y = pd.DataFrame(data_filtered, columns=['Survived'])
clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)
importances = clf.feature_importances_
