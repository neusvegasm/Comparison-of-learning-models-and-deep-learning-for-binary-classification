import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats

outliers = False
scale = False
irrelevant = False
smaller = True

df = pd.read_csv('data.csv')

df.drop('id', axis=1, inplace=True)


# Convertir la columna 'diagnosis' a valores numéricos
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

if outliers:
    # REMOVING OUTLIERS using z-score
    z_scores = stats.zscore(df)
    df = df[(np.abs(z_scores) < 3).all(axis=1)]

data = df

if smaller:
      data = data.sample(n=100, random_state=42)

# Separar los datos en características (X) y variable objetivo (y)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

if scale:
    # ESCALAR LOS DATOS
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

if irrelevant:
    # Agregar un atributo IRRELEVANTE
    X_train['irrelevant_attribute'] = np.random.random(size=len(X_train))
    X_test['irrelevant_attribute'] = np.random.random(size=len(X_test))


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
