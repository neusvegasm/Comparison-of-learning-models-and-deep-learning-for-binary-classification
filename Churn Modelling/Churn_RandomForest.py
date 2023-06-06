import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


rf_model = RandomForestClassifier(random_state=2020)
rf_model.fit(X_train, y_train)

pred_test = rf_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred_test) * 100, '%')

'''
Podem modificar els paràmetres:

- n_estimators: the number of trees in the forest.
- max depth: the maximum depth of the tree.
'''

'''
random_grid = {'max_depth': [1, 5, 10, 15],
               'n_estimators': [100, 200, 300, 400, 500, 600]}
rf_random = RandomizedSearchCV(rf_model, random_grid, n_iter=50, cv=5, random_state=2020)
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)
'''

# Un cop tenim els resultats podem tornar a buscar els paràmetres óptims centrant-nos en un rang més concret:

'''
param_dist = {"max_depth": [15, 16, 17, 18, 19],
              "n_estimators": [600, 650, 700, 750, 800]}
rf_cv = GridSearchCV(rf_model, param_dist, cv=5)
rf_cv.fit(X_train, y_train)
print(rf_cv.best_params_)
'''
# Model utilitzant el smillors parámetres:

rf_new = RandomForestClassifier(n_estimators=1500, max_depth=16, random_state=2020)
rf_new.fit(X_train, y_train)
y_pred_new = rf_new.predict(X_test)
print('New accuracy:', accuracy_score(y_test, y_pred_new) * 100, '%')
