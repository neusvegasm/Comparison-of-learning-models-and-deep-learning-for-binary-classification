import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
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

