import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats

df = pd.read_csv('winequality-red.csv')

#Establim el valor per separar en 2 classes
df.loc[df['quality'] < 5, 'quality'] = 0
df.loc[df['quality'] >= 5, 'quality'] = 1

##ELIMINEM OUTLIERS Y REESCALEM EL DATASET (Z-SCORE)

print(df.shape)
z = np.abs(stats.zscore(df.iloc[:, :-1]))
df = df[(z < 3).all(axis=1)]
data = np.abs(stats.zscore(df))
print(df.shape)



#TORNEM A REESCALAR ELS OUTPUTS A 0 I 1
print(data['quality'].unique())
data.loc[data['quality'] < 5, 'quality'] = 0
data.loc[data['quality'] >= 5, 'quality'] = 1


# Separem les dades en train i test
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)

model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)],
          verbose=20, eval_metric='logloss')

pred_test = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, pred_test) * 100, '%')
