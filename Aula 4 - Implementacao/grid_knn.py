# -*- coding: utf-8 -*-
"""
Aula 4 - grid com knn
"""
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Train Test Split
trainX, testX, trainY, testY = train_test_split(dataset, label, test_size=0.20,random_state=42)

#List Hyperparameters that we want to tune.
n_neighbors = list(range(1,30))

grid = GridSearchCV(KNeighborsClassifier(), n_neighbors=n_neighbors , cv=5, refit = True, n_jobs=-1, verbose = 2)

# fitting the model for grid search
grid.fit(trainX, trainY)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

model = KNeighborsClassifier(leaf_size=1, n_neighbors=11, p=1)
model.fit(trainX, trainY)  

# save the model to disk
filename = 'model_knn.sav'
pickle.dump(model, open(filename, 'wb'))

predictions = model.predict(testX)
print(classification_report(testY, predictions))