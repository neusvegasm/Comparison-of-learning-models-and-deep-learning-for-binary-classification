import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats



outliers = False
scale = False
irrelevant = False
smaller = False

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