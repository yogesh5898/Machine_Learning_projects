import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

""" Loading dataset """
dataset = pd.read_csv('dataset.csv')

""" Summarize the dataset """
print(dataset.head(5))
print(dataset.shape)

""" Segregating data into X and Y """
X = dataset.drop('price', axis='columns')
Y = dataset.price

""" Visualize a dataset """
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(X, Y, color='red', marker='*')
plt.show()

""" Training dataset and Linera Regression """
model = LinearRegression()
model.fit(X, Y)

""" Predicting Price based on input Area """
x = 400000
land_area_in_sqrt = [[x]]
result = model.predict(land_area_in_sqrt)
print(result)

""" Verify based on theory calculation """
m = model.coef_
print(m)

b = model.intercept_
print(b)

y = m*x + b
print('The price of {0} square feet land is : {1}'.format(x, y[0]))