#importing necessary modules
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
Xtrain = [[182, 80, 34], [176, 70, 33], [161, 60, 28], [154, 55, 27], [166, 63, 30], [189, 90, 36], [175, 63, 28], [177, 71, 30], [159, 52, 27], [171, 72, 32], [181, 85, 34]]

Ytrain = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

Xval = [[163, 62, 28], [182, 80, 35], [150, 50, 24], [160, 57, 27], [175, 62, 30], [183, 67, 32], [177, 64, 29], [164, 62, 29], [157, 53, 23], [170, 73, 32], [169, 59, 29]]

Yval = ['female', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'female']

# initializing the ML models
knn = KNeighborsClassifier()
perceptron = Perceptron()

# Fitting the models
knn.fit(Xtrain, Ytrain)
perceptron.fit(Xtrain, Ytrain)

# Testing using our input data
pred_knn = knn.predict(Xval)
acc_knn = accuracy_score(Yval, pred_knn) * 100
print(f'Accuracy for knn: {acc_knn}')

pred_perceptron = perceptron.predict(Xval)
acc_perceptron = accuracy_score(Yval, pred_perceptron) * 100
print(f'Accuracy for perceptron: {acc_perceptron}')

# The best classifier out of the two models
index = np.argmax([acc_knn, acc_perceptron])
#argmax function assigns the index of the maximum value to the variable
classifiers = {0: 'KNN', 1:'PER'}
print(f'Best gender classifier is {classifiers[index]}')
