# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:19:16 2023

@author: Mouhammed Dbs
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# read data
df = pd.read_csv('persons_dataset.csv')
# convert DataFram to NumPy Array
data = df.values

X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=42)

# print(X_train[0])
# print('----------------')
# print(y_train)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # Create Model
model = Sequential()
model.add(Dense(126, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(38, activation='softmax'))

# # Train Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size=40)
model.save('model.h5')
# # Model Evaluation
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))