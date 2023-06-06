import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.impute import KNNImputer

smaller = False
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
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

if scale:
    # Reescalem les variables per a que tinguin mitjana 0 i desviació 1
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


#Entrenament
lr = LogisticRegression(random_state=40)
lr.fit(X_train, Y_train)


train_accuracy = lr.score(X_train, Y_train)*100
test_accuracy = lr.score(X_test, Y_test)*100
print('One-vs-rest', '-'*35,
      'Accuracy in Train Group   : {:.2f} %'.format(train_accuracy),
      'Accuracy in Test  Group   : {:.2f} %'.format(test_accuracy), sep='\n')

pred_test = lr.predict(X_test)

confusion_matrix_test = confusion_matrix(Y_test, pred_test)

TN = confusion_matrix_test[0][0]
TP = confusion_matrix_test[1][1]
FP = confusion_matrix_test[0][1]
FN = confusion_matrix_test[1][0]

print("(Total) True Negative       :", TN)
print("(Total) True Positive       :", TP)
print("(Total) False Positive   :", FP)
print("(Total) False Negative   :", FN)