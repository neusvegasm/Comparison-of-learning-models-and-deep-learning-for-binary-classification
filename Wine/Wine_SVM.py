import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

outliers = True
scale = True
irrelevant = False
redundant = False

df = pd.read_csv('winequality-red.csv')


if outliers:
    # REMOVING OUTLIERS using z-score
    z_scores = stats.zscore(df)
    df = df[(np.abs(z_scores) < 3).all(axis=1)]



#Establim el valor per separar en 2 classes
df.loc[df['quality'] < 6, 'quality'] = 0
df.loc[df['quality'] >= 6, 'quality'] = 1
data = df



# Separem les dades en train i test
X, y = data.iloc[:, :-1], data.iloc[:, -1]


if scale:
    # SCALE DATA
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)

if irrelevant:
    # Add a IRRELEVANT ATTRIBUTE
    X_train['irrelevant_attribute'] = np.random.random(size=len(X_train))
    X_test['irrelevant_attribute'] = np.random.random(size=len(X_test))


if redundant:
    # Add a REDUNDANT ATTRIBUTE
    X_train['redundant_attribute'] = X_train['residual sugar']
    X_test['redundant_attribute'] = X_test['residual sugar']



svc = SVC(random_state=2020)
svc.fit(X_train, y_train)

pred_test = svc.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred_test) * 100, '%')

confusion_matrix_test = confusion_matrix(y_test, pred_test)

TN = confusion_matrix_test[0][0]
TP = confusion_matrix_test[1][1]
FP = confusion_matrix_test[0][1]
FN = confusion_matrix_test[1][0]

print("(Total) True Negative       :", TN)
print("(Total) True Positive       :", TP)
print("(Total) False Positive   :", FP)
print("(Total) False Negative   :", FN)

'''

random_grid = {"C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
svc_random = RandomizedSearchCV(svc, random_grid, cv=5, random_state=2020)
svc_random.fit(X_train, y_train)
print('Millor parámetre seleccionat:', svc_random.best_params_)

svc_new = SVC(C=1000, random_state=2020)
svc_new.fit(X_train, y_train)
y_pred_new = svc_new.predict(X_test)
print('New accuracy:', accuracy_score(y_test, y_pred_new)*100, '%')

'''

'''
Podem modificar els paràmetres:

- C: regularization parameter
- kernel: ‘linear,’ ‘poly,’ ‘rbf.’
'''

'''
param_dist = {'C': [1, 100, 1000],
              'kernel': ['linear', 'rbf', 'poly']}
svc_cv = GridSearchCV(svc, param_dist, cv=10)
svc_cv.fit(X_train, y_train)
print(svc_cv.best_params_)

svc_new = SVC(C=1.3, kernel="rbf", random_state=2020)
svc_new.fit(X_train, y_train)
y_pred_new = svc_new.predict(X_test)
print(accuracy_score(y_test, y_pred_new))
'''