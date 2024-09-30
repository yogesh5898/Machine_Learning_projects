import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


""" Loading dataset """
dataset = pd.read_csv('digit.csv')
print(dataset.head())

print(dataset.shape)

""" Segregating X and Y """
X = dataset.iloc[:, 1:]
print(X.head())

Y = dataset.iloc[:, 0]
print(Y.head())

""" splitting dataset into Train an Test """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Training """
model = RandomForestClassifier()
model.fit(X_train, Y_train)

""" prediction """
y_prediction = model.predict(X_test)

""" Accuracy """
print("Accuracy of the model {0}%".format(accuracy_score(y_prediction, Y_test)* 100))

index = 687
print("Predicted ----> " + str(model.predict(X_test)[index]))
plt.axis('off')
plt.imshow(X_test.iloc[index].values.reshape((28, 28)), cmap='grey')
plt.show()