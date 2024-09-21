from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

""" Loading dataset """
dataset = load_iris()

""" Summarize the data """
print(dataset.data)
print(dataset.target)

print(dataset.data.shape)

""" Segregating X and Y 
X =  sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) 
Y = Setosa  Versicolor  virginica """

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
Y = dataset.target

""" Splitting dataset into Train and test """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Finding Max_Depth_Value """
accuracy = []

for i in range(1, 10):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    score = accuracy_score(Y_test, prediction)
    accuracy.append(score)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_depth')
plt.xlabel('Predict')
plt.ylabel('Score')
#plt.show()

""" Training """
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
model.fit(X_train, Y_train)

""" Prediction """
y_prediction = model.predict(X_test)
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), Y_test.reshape(len(Y_test), 1)), 1))

""" Accuracy """
print("Accuracy Score of the model : {0}%".format(accuracy_score(Y_test, y_prediction)* 100))

""" Predicting new data 
X =  sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) 
Y = Setosa  Versicolor  virginica """

sepal_length = float(input("Enter the Sepal_length : "))
sepal_width = float(input("Enter the Sepal_width : "))
petal_length = float(input("Enter the Petal_length : "))
petal_width = float(input("Enter the Petal_width : "))

new_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
result = model.predict(new_data)
print(result)

if result == 1:
    print('It is Setosa')
elif result == 2:
    print('It is Versicolor')
else:
    print('It is virginica')