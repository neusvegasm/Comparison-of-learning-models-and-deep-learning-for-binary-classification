import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Leer los datos del archivo CSV
data = pd.read_csv('input_2.csv')

## PREPROCESAMIENTO

# Eliminar las columnas que no aportan información
data.drop(['ID_PERSONA', 'ID','A','SLABORAL','M_ALTA'], axis=1, inplace=True)

# Separar las características (X) de la variable objetivo (Y)
X = data.drop(['BAIXA_AFILIACIO'], axis=1)
Y = data['BAIXA_AFILIACIO'].to_numpy()
Y = np.expand_dims(Y, axis=1)  # Añadir una dimensión a Y

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizar los datos de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Crear un conjunto de datos personalizado para PyTorch
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Crear objetos DataLoader para cargar los datos en lotes durante el entrenamiento
batch_size = 40
train_dataset = CustomDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Definir el número de características de entrada para el modelo
input_size = X.shape[1]

# Definir el modelo de la red neuronal
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

## ENTRENAMIENTO

# Definir la función de pérdida y el optimizador
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Definir el número de épocas de entrenamiento
num_epochs = 100

# Poner el modelo en modo de entrenamiento
model.train()

# Ciclo de entrenamiento
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")



# Poner el modelo en modo de evaluación
model.eval()

# Evaluar el modelo en los datos de prueba
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    predictions = (model(X_test_tensor) >= 0.5).float()
    accuracy = (predictions == Y_test_tensor).float().mean()
    print(f"Accuracy on test data: {accuracy.item():.4f}")


# Obtener las predicciones del modelo en los datos de prueba
with torch.no_grad():
    predictions = (model(X_test_tensor) >= 0.5).float()

# Convertir las predicciones y las etiquetas a numpy arrays
predictions = predictions.numpy().flatten()
Y_test = Y_test.flatten()

# Calcular la matriz de confusión
confusion_mat = confusion_matrix(Y_test, predictions)

# Mostrar la matriz de confusión
print("Matriz de Confusión:")
print(confusion_mat)