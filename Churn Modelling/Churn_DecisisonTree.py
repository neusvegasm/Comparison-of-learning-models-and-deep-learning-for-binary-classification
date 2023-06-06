import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler


smaller = False
balanced = True
irrelevant = False
scale = False
redundant = False
missing_values = False

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


if missing_values:
    # Set probability of generating a NaN for each cell
    nan_prob = 0.2
    # Generate a random mask with the same shape as the DataFrame
    mask = np.random.choice([False, True], size=X.shape, p=[1 - nan_prob, nan_prob])
    # Replace values with NaN where the mask is True
    X[mask] = np.nan
    # Fill NaN values with the mean of each column
    X = X.fillna(X.mean())

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


'''
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
'''