import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
