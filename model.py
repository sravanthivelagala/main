import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

### Import Datset
df = pd.read_csv("data.csv")
# we change the class values (at the column number 2) from B to 0 and from M to 1
df.iloc[:, 1].replace('B', 0,inplace=True)
df.iloc[:, 1].replace('M', 1,inplace=True)

### Splitting Data

X = df[['texture_mean','area_mean','concavity_mean','area_se','concavity_se','fractal_dimension_se','smoothness_worst','concavity_worst', 'symmetry_worst','fractal_dimension_worst']]
y = df['diagnosis']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

#### Data Preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

print("Confusion Matrix : \n\n" , confusion_matrix(y_pred_dt, y_test))

print("Classification Report : \n\n" , classification_report(y_pred_dt, y_test),"\n")

pickle.dump(dt_classifier, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)
