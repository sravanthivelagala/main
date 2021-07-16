import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data.csv')

df.iloc[:,1].replace('B', 0,inplace=True)
df.iloc[:,1].replace('M', 1,inplace=True)

X = df[['texture_mean','area_mean','concavity_mean','area_se','concavity_se','fractal_dimension_se','smoothness_worst','concavity_worst', 'symmetry_worst','fractal_dimension_worst']]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

print('*********RadomForestClassifier accuracy**********')

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred_rfc = classifier.predict(X_test)
accuarcy_rfc = accuracy_score(y_test, y_pred_rfc)
print('for trianing data set', accuarcy_rfc)

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_test, y_test)
y_pred_rfc = classifier.predict(X_test)
accuarcy_rfc = accuracy_score(y_test, y_pred_rfc)
print('for testing data set', accuarcy_rfc)

print('*********DecisionTreeClassifier accuracy**********')

dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuarcy_dt = accuracy_score(y_test, y_pred_dt)
print('for training data set', accuarcy_dt)

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_test, y_test)
y_pred_dt = dt_classifier.predict(X_test)
accuarcy_dt=accuracy_score(y_test, y_pred_dt)
print('for testing data set', accuarcy_dt)

print('*********Naivebayes accuracy**********')
nb = GaussianNB()
nb.fit(x_train, y_train)
print('for training data set', nb.score(x_train, y_train))
print('for testing data set', nb.score(x_test, y_test))

print('*********KNeighborsClassifier accuracy**********')
knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)
print('for training data set',knn.score(X_train, y_train))
print('for testing data set', knn.score(X_test, y_test))

print(classification_report(y_test, y_pred_dt))

crossval_score = cross_val_score(dt_classifier, X_train, y_train)
print('crossval_score', crossval_score)
print('mean crossval_score', crossval_score.mean())
