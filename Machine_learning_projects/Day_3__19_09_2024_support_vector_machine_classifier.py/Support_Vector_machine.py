import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

""" Loading dataset """
dataset = load_digits()

""" Summarize a dataset """
print(dataset.data)
print(dataset.target)   # 0 to 9

print(dataset.data.shape)
print(dataset.images.shape)

data_Image_Length = len(dataset.images)
print(data_Image_Length)

""" Visualize the dataset """
n = 5
plt.gray()
plt.matshow(dataset.images[n])
#plt.show()

print(dataset.images[n])


""" Segregating X and Y """
X = dataset.images.reshape((data_Image_Length, -1))
Y = dataset.target


""" Splitting dataset into train and test """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Training model 
kernel = 'linear', 'rbf'
gamma = 0.001, c=0.1 
"""
model = svm.SVC()  # If we didn't pass any parameter it will take default
model.fit(X_train, Y_train)

""" Prediction """
y_prediction = model.predict(X_test)
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), Y_test.reshape(len(Y_test), 1)), 1))


""" Accuracy """
print("Accuracy of the model : {0}%".format(accuracy_score(Y_test, y_prediction)* 100))


""" prediction what digit from test data """
n=7

result = model.predict(dataset.images[n].reshape((1, -1)))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print('result :', result, sep='\n')
plt.axis('off')
plt.title('%i' %result)
plt.show()
