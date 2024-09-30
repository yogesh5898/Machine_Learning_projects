import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


""" Loading dataset """
dataset = pd.read_csv('data.csv')

""" Summarize the dataset """
print(dataset.head())
print(dataset.shape)
print(dataset.dtypes)
print(dataset.info())  # to check null values present in dataset

""" Mapping diagnosis column to 0 to 1 """
dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

""" Segregating X and Y (starts with 0)"""
X = dataset.iloc[:, 2:31].values
Y = dataset.iloc[:, 1].values

""" Splitting data into Train and Test """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Feature Scaling """
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)

""" Validating some ML Algorithms by its accuracy """

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Cart', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print('%s: %f' % (name, cv_results.mean()))

plt.ylim(0.90, 0.99)
plt.bar(names, res, color='maroon', width=0.6)

plt.title('Algorithm Comparison')
plt.show()

""" Will Train logistic Regression """
model = LogisticRegression()
model.fit(X_train, Y_train)

""" predicting our new data """
value = [[16.02,23.24,102.7,797.8,0.08206,0.06669,0.03299,0.03323,0.1528,0.05697,0.3795,1.187,2.466,40.51,0.004029,0.009269,
          0.01101,0.007591,0.0146,0.003042,19.19,33.88,123.8,1150,0.1181,0.1551,0.1459,0.09975,0.2948]]

value = sc.transform(value)
y_prediction = model.predict(value)

if y_prediction == 1:
    print('Diagnosis : M')
else:
    print('Diagnosis : B')