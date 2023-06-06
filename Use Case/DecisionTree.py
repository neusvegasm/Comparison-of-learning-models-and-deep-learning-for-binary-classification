import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt



# Cargar los datos
data = pd.read_csv('input_2.csv')

# Eliminar las columnas no informativas
data.drop([ 'ID_PERSONA', 'ID','A'], axis=1, inplace=True)

# Separar las características de la variable objetivo
X = data.drop(['BAIXA_AFILIACIO'], axis=1)
y = data['BAIXA_AFILIACIO']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo de árbol de decisiones
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print("Precisión del modelo:", accuracy)

importances = model.feature_importances_
feature_names = X.columns

# Crear un dataframe con las importancias de las características
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Imprimir las características más influyentes
print(feature_importances)

# Calcular la matriz de confusión
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Matriz de Confusión:")
print(cm)


# Visualizar el árbol de decisiones
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No baja', 'Baja'], filled=True, rounded=True)
plt.show()
