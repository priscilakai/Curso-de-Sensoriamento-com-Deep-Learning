# -*- coding: utf-8 -*-
"""
Aula 4 - grid com svm
"""
import pickle
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

# Train Test Split
trainX, testX, trainY, testY = train_test_split(dataset, label, test_size=0.20,random_state=42)

#List Hyperparameters that we want to tune.
parameters = {
            'kernel':('linear', 'polynomial','sigmoid','rbf'), 
            'C':[0.01, 0.1, 1, 10, 100, 1000]
            }

grid = GridSearchCV(svm.SVC(), parameters, cv=5, refit = True, n_jobs=-1, verbose = 2)

# fitting the model for grid search
grid.fit(trainX, trainY)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

model = svm.SVC(C=1000, kernel='rbf')
model.fit(trainX, trainY)

# save the model to disk
filename = 'model_svm.sav'
pickle.dump(model, open(filename, 'wb'))

predictions = model.predict(testX)
print(classification_report(testY, predictions))
