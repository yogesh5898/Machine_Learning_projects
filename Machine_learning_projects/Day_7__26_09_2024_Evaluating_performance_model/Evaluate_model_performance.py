import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve


""" Loading dataset """
dataset = pd.read_csv('DigitalAd_dataset.csv')

""" Summarize the data """
print(dataset.shape)
print(dataset.head())

""" Segregating X and Y """
X = dataset.iloc[:, :-1].values  # Age and Salary
Y = dataset.iloc[:, -1].values   # Status

""" Splitting dataset into Train and Test """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Feature Scaling """
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

""" Training """
model = LogisticRegression(random_state=0)
model.fit(X_train, Y_train)

""" Prediction for all test data """
y_prediction = model.predict(X_test)
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), Y_test.reshape(len(Y_test), 1)), 1))

""" Model Evaluation """


""" Confusion Matrix """
cm = confusion_matrix(Y_test, y_prediction)
print("Confusion Matrix output : ", sep='\n')
print(cm)


""" Accuracy Score """
print("Accuracy of the model {0}%".format(accuracy_score(Y_test, y_prediction) * 100))


""" Receive Operating Curve - ROC curve """
nsProbability = [0 for _ in range(len(Y_test))]   # Random line
lsProbability = model.predict_proba(X_test)       # Logistic Regression Probability

""" Keeping probability of positive outcome only """
lsProbability = lsProbability[:, 1]

""" Calculate Scores """
nsAUC = roc_auc_score(Y_test, nsProbability)    # No skill AUC curve
lrAUC = roc_auc_score(Y_test, lsProbability)    # Logistic skill AUC curve

""" Summarize Scores """
print('No skill : ROC AUC= %.3f' % (nsAUC*100))
print('Logistic skill : ROC AUC= %.3f' % (lrAUC*100))

""" Calculate ROC Curve """
nsFP, nsTP, _ = roc_curve(Y_test, nsProbability)
lrFP, lrTP, _ = roc_curve(Y_test, lsProbability)

""" Plot the ROC curve of the model """
plt.plot(nsFP, nsTP, linestyle='--', label='No skill')
plt.plot(lrFP, lrTP, marker='*', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

""" Show the legend """
plt.legend()
#plt.show()


""" K_fold Cross Validation """
k_fold = KFold(n_splits=10)
result = cross_val_score(model, X, Y, cv=k_fold)
print("Cross validation Score : %.2f%%" % (result.mean()*100.0) )


""" Statified K_fold Cross Valiation """
sk_fold = StratifiedKFold(n_splits=3)
model_skfold = LogisticRegression()
result_skfold = cross_val_score(model_skfold, X, Y, cv=sk_fold)
print("Stratified Cross validation Score : %.2f%%" % (result_skfold.mean()*100.0))


