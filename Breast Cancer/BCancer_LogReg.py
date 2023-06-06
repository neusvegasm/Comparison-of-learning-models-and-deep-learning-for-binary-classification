import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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


lr = LogisticRegression(random_state=40)
lr.fit(X_train, y_train)

train_accuracy = lr.score(X_train, y_train) * 100
test_accuracy = lr.score(X_test, y_test) * 100
print('One-vs-rest', '-' * 35,
      'Accuracy in Train Group   : {:.2f} %'.format(train_accuracy),
      'Accuracy in Test  Group   : {:.2f} %'.format(test_accuracy), sep='\n')

pred_test = lr.predict(X_test)

confusion_matrix_test = confusion_matrix(y_test, pred_test)

TN = confusion_matrix_test[0][0]
TP = confusion_matrix_test[1][1]
FP = confusion_matrix_test[0][1]
FN = confusion_matrix_test[1][0]

print("(Total) True Negative       :", TN)
print("(Total) True Positive       :", TP)
print("(Total) False Positive   :", FP)
print("(Total) False Negative   :", FN)
