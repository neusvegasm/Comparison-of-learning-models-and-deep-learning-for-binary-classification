import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)

if irrelevant:
    # Add a IRRELEVANT ATTRIBUTE
    X_train['irrelevant_attribute'] = np.random.random(size=len(X_train))
    X_test['irrelevant_attribute'] = np.random.random(size=len(X_test))


if redundant:
    # Add a REDUNDANT ATTRIBUTE
    X_train['redundant_attribute'] = X_train['residual sugar']
    X_test['redundant_attribute'] = X_test['residual sugar']

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

if scale:
    #reescalem les variables per a que tinguin mitjana 0 i desviació 1
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

# Transformem a format tensor de pytorch:

class Data(Dataset):
    def __init__(self, X_train, y_train):
        # Necesitem transformar float64 a float32 si no
        # ens apareixerà el següent error
        # RuntimeError: expected scalar type Double but found Float
        self.X = torch.from_numpy(X_train.astype(np.float32))
        # ara hem de convertir el float64 a Long sino
        # ens apareixerà el següent error
        # RuntimeError: expected scalar type Long but found Float
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


traindata = Data(X_train, Y_train)
# ara es pot accedir a les dades:
print(traindata[25:34])

# A trevés del següent codi creem i definim els batch que utilitzarem per entrenar la nostra xarxa neuronal.
# En aquets cas la mida de cadascun será de 4.

batch_size = 50
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)

# Creem els parámetres de la xarxa:

input_dim = X_train.shape[1]  # número de variables del dataset
hidden_dim_1 = 16  # parámetres capa oculta 1
hidden_dim_2 = 32  # parámetres capa oculta 2
output_dim = 2  # número de classes


# Definim la classe de la xarxa neuronal

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim_1)
        self.linear2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.linear3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))  # aquí podem cambiar a relu, sigmoide, etc
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


clf = Network()
print(clf.parameters)  # 11 parámetres de input i 2 de output

# funció de pérdua i optimitzador:

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)

# Entrenament ( podem canviar el número de epochs):

epochs = 100

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # set optimizer to zero grad to remove previous epoch gradients
        optimizer.zero_grad()
        # forward propagation
        outputs = clf(inputs)
        loss = criterion(outputs, labels)
        # backward propagation
        loss.backward()
        # optimize
        optimizer.step()
        running_loss += loss.item()
        # display statistics
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')

# Guardar el model entrenat
PATH = './mymodel.pth'
torch.save(clf.state_dict(), PATH)

# Carregar el model
clf = Network()
clf.load_state_dict(torch.load(PATH))
'''
# Output
<All keys matched successfully>
'''

# Testejar el model entrenat:

testdata = Data(X_test, Y_test)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)
# agafem un mini batch
dataiter = iter(testloader)
inputs, labels = dataiter.__next__()

correct, total = 0, 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        # calculem el output de la xarxa neuronal
        outputs = clf(inputs)
        # guardem prediccions
        __, predicted = torch.max(outputs.data, 1)
        # actualitzem resultats
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy de la xarxa amb {len(testdata)} dades de test: {100 * correct / total} %')
