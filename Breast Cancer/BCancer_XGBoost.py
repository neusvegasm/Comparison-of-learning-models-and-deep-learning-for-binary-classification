import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
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



clf_xgb = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10,
                            seed=123)
clf_xgb.fit(X_train, y_train)
preds = clf_xgb.predict(X_test)
accuracy_xgb = float(np.sum(preds == y_test))/y_test.shape[0]
print('Accuracy de XGBoost: ', accuracy_xgb*100, '%')

confusion_matrix_test = confusion_matrix(y_test, preds)


TN = confusion_matrix_test[0][0]
TP = confusion_matrix_test[1][1]
FP = confusion_matrix_test[0][1]
FN = confusion_matrix_test[1][0]

print("(Total) True Negative       :", TN)
print("(Total) True Positive       :", TP)
print("(Total) False Positive   :", FP)
print("(Total) False Negative   :", FN)

