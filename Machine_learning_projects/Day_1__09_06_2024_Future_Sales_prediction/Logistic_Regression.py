import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score

dataset = pd.read_csv('DigitalAd_dataset.csv')
print(dataset.head())

"""Gives total no of rows and columns"""
print(dataset.shape)

""" Segregating X and Y 
    X - Age and Salary  and Y - Status """

x = dataset.iloc[:, :-1]  # [rows, columns] [: means all rows, :-1 means except last column]
y = dataset.iloc[:, -1]  # It will consider only last column

"""Splitting dataset into train and test 

X_train - Input 80% data  (feature)
Y_train - Output 80% data (Label)
X_test  - Input 20% data
Y_test  - Output 20% data

random_state - Will take data in shuffled way 
"""

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

"""Feature Scaling
Fit_transform - Fit method is calculating mean and variance of each of the data present in our data
transform     - It transforming all the features using the respective mean and variance
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Training data should be efficient
X_test  = sc.transform(X_test)

""" Selecting Algorithm """
model = LogisticRegression()

"""" Training our model """
model.fit(X_train, Y_train)

""" Validation """
y_prediction = model.predict(X_test)
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), Y_test.values.reshape(len(Y_test), 1)), 1))

print("Accuracy of the model : {0}%".format(accuracy_score(Y_test, y_prediction)*100))

""" Prediction """
age = int(input("Enter the new Customer Age : "))
salary = int(input("Enter the Customer Salary : "))

New_customer = pd.DataFrame([[age, salary]], columns=x.columns)

Result = model.predict(sc.transform(New_customer))
print(Result)

if Result == 1:
    print('Customer will buy')
else:
    print("Customer will not buy")