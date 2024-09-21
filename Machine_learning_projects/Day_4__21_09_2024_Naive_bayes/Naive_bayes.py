import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

""" Loading dataset """
dataset = pd.read_csv('titanicsurvival.csv')

""" Summarize the dataset """
print(dataset.head())
print(dataset.shape)

""" Mapping text data to binary values """
dataset['Gender'] = dataset['Gender'].map({'male': 1, 'female': 0}).astype(int)
print(dataset.head())

""" Segregating X and Y """
X = dataset.drop(['Survived'], axis='columns')  # axis = 'columns' which we are planned to drop
Y = dataset['Survived']

""" Finding and removing NAn data from our feature X """
print(X.columns[X.isna().any()])
X['Age'] = X['Age'].fillna(X['Age'].mean())


""" Splitting dataset into train and test """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Training """
model = GaussianNB()
model.fit(X_train, Y_train)

""" Prediction """
y_prediction = model.predict(X_test)
print(np.column_stack((y_prediction, Y_test)))

""" Accuracy """
print('Accuracy score of the model {0}%'.format(accuracy_score(y_prediction, Y_test) * 100))

""" Prediction of new customer """
passenger_Class_No = int(input("Passenger_Class_Number :"))
gender = int(input("Enter the Gender : "))
age = int(input("Enter the Age : "))
fare = float(input("Enter the Fare : "))  # it will accept both int and float

person = pd.DataFrame([[passenger_Class_No, gender, age, fare]], columns=X.columns)
result = model.predict(person)
print(result)

if result == 1:
    print("Person might be survived")
else:
    print("Person will not be survived")