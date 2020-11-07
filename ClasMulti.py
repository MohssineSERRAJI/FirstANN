# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:08:15 2020

@author: SERRAJI Mohssine
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
    model.add(Dense(12, input_dim=7, kernel_initializer="uniform", activation="relu" ))
    model.add(Dense(23, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(15, kernel_initializer="uniform", activation="relu"))
    #I use softmax because we are in multiclass problem
    model.add(Dense(3, kernel_initializer="uniform", activation="softmax"))
    
    #to configure the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[ 'accuracy'])

    return model


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

dataset = load_data("bases/seeds.csv")

X = dataset[:,:-1]
Y = dataset[:,-1]


model = KerasClassifier(build_fn=define_model , epochs= 150, batch_size=10, verbose=1)
#cross validation with k fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

#applie the cross validation on the model like fit

results = cross_val_score(model, X, Y, cv=kfold)

#display the means off scores on 10-folds

print(results.mean())
