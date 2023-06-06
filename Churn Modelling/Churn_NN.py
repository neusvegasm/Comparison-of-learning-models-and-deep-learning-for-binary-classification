import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer

smaller = False
missing_values = False
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
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

if scale:
    # Reescalem les variables per a que tinguin mitjana 0 i desviació 1
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train.astype(np.float64))
Y_train_tensor = torch.Tensor(Y_train).long()
X_test_tensor = torch.Tensor(X_test.astype(np.float64))
Y_test_tensor = torch.Tensor(Y_test).long()
# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.features)

# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first hidden layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after the second hidden layer
        x = self.fc3(x)
        return x

# Initialize the neural network
input_size = X_train.shape[1]
net = Net(input_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Create DataLoader objects for training and testing
batch_size = 60
train_dataset = CustomDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the neural network
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Print the loss after each epoch
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

# Convert the test data to PyTorch tensors
X_test_tensor = torch.Tensor(X_test.astype(np.float32))
Y_test_tensor = torch.Tensor(Y_test).long()

# Evaluate the neural network on the test data
with torch.no_grad():
    net.eval()
    outputs = net(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == Y_test_tensor).sum().item() / len(Y_test_tensor)
    print("Accuracy:", accuracy*100)