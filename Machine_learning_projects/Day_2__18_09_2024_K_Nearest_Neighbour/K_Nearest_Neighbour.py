import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('salary.csv')
print(dataset.shape)

""" Mapping salary data to binary value """
income_set = set(dataset['income'])
dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
# print(dataset.head())

""" Segregating X and Y """
X = dataset.iloc[:, :-1]    # Age, Education_num, Capital_gain, Hours_per_week
Y = dataset.iloc[:, -1]     # Income

""" Model Selection """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Feature Scaling """
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

""" Finding the best K value """
error = []

for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, Y_train)
    prediction_i = model.predict(X_test)
    error.append(prediction_i != Y_test)

plt.figure(figsize=(12, 8))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error rate K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')

""" Training Model
n_neighbors=2 is the K value
metric='minkowski' 
p=2 is Euclidean distance
"""
model = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
model.fit(X_train, Y_train)

""" Validation """
y_prediction = model.predict(X_test)
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), Y_test.values.reshape(len(Y_test), 1)), 1))

print("Accuracy score of the mode {0}%".format(accuracy_score(Y_test, y_prediction)*100))

""" prediction new customer data """
age = int(input('Enter the Age : '))
education = int(input('Enter the Education : '))
capital_gain = int(input('Enter the Capital Gain : '))
work_in_hour = int(input('Enter the work/hr :'))

new_employee = pd.DataFrame([[age, education, capital_gain, work_in_hour]], columns=X.columns)
result = model.predict(sc.transform(new_employee))
print(result)

if result == 1:
    print("Employee salary is above 50K")
else:
    print("Employee salary is less than 50K")