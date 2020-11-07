# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:44:59 2020

@author: Mohssine SERRAJI
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

seed = 7
np.random.seed(seed)

def load_data(filepath):
    dataset = np.loadtxt(filepath, delimiter=",")
    #print(type(dataset))
    return dataset
    


def define_model():
    #create the model using Sequential class
    model = Sequential()
    
    #add layers to the model with nb_neurals and activation function
    model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu" ))
    model.add(Dense(23, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(15, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
    
    #to configure the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[ 'accuracy'])

    return model



#k fold cross validation i use 10-fold

#method number 1
from sklearn.model_selection import StratifiedKFold

dataset = load_data("bases/diabetes.csv")

X = dataset[:,:-1]
Y = dataset[:,-1]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores = []

for train, test in kfold.split(X, Y):
    model = define_model()
    #Start the training
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


print("the second part !!!!")

#method number 1
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

model = KerasClassifier(build_fn=define_model , epochs= 150, batch_size=10, verbose=0)
#cross validation with k fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

#applie the cross validation on the model like fit

results = cross_val_score(model, X, Y, cv=kfold)

#display the means off scores on 10-folds

print(results.mean())
