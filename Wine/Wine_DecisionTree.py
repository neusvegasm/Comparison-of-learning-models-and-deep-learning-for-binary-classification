import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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


dt = DecisionTreeClassifier(random_state=2020)
dt.fit(X_train, y_train)

pred_test = dt.predict(X_test)
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

# optimitzar parámetres:

param_dist = {"max_depth": range(1, 6),
              "max_features": range(1, 10),
              "criterion": ["gini", "entropy"]}
dt_cv = GridSearchCV(dt, param_dist, cv=5)
dt_cv.fit(X_train, y_train)
print('Millor parámetre seleccionat:', dt_cv.best_params_)

dt_new = DecisionTreeClassifier(criterion="entropy",
                                max_depth=5,
                                max_features=8,
                                random_state=2020)
dt_new.fit(X_train, y_train)
y_pred_new = dt_new.predict(X_test)
print('New accuracy:', accuracy_score(y_test, y_pred_new)*100, '%')
