import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer

smaller = False
missing_values = True
balanced = False
irrelevant = False
scale = False
redundant = False

data = pd.read_csv('Churn_Modelling.csv', index_col='RowNumber')





## PREPROCESSAMENT

# Eliminem les columnes que no ens aporten informació:
data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# Transformem variables categòriques a dummys (millor això que passar-les a numèriques ja que té més sentit)
data = pd.get_dummies(prefix='Geo', data=data, columns=['Geography'])
data = data.replace(to_replace={'Gender': {'Female': 1, 'Male': 0}})
print(data['IsActiveMember'])

if smaller:
      data = data.sample(n=1000, random_state=42)


if missing_values:
    # Perform k-nearest neighbors imputation for missing values
    imputer = KNNImputer(n_neighbors=5)  # Adjust the number of neighbors as needed
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

## ENTRENAMENT

X = data.drop(['Exited'], axis=1)
Y = data.Exited

if balanced:
      undersampler = RandomUnderSampler(random_state=42)
      X, Y = undersampler.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


if irrelevant:
    # Add a IRRELEVANT ATTRIBUTE
    X_train['irrelevant_attribute'] = np.random.random(size=len(X_train))
    X_test['irrelevant_attribute'] = np.random.random(size=len(X_test))


if redundant:
    # Add a REDUNDANT ATTRIBUTE
    X_train['redundant_attribute'] = X_train['NumOfProducts']
    X_test['redundant_attribute'] = X_test['NumOfProducts']

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = Y_train.to_numpy()
y_test = Y_test.to_numpy()

if scale:
    # Reescalem les variables per a que tinguin mitjana 0 i desviació 1
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


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