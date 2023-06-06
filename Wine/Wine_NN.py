"""
En aquest exemple es crea un model lineal amb Pytorch per tan de predir la qualitat d'un vi segons les seves característiques
"""




import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter('venv/Scripts/runs/wine1')


df = pd.read_csv('winequality-red.csv')
df.loc[df['quality'] < 6, 'quality'] = 0
df.loc[df['quality'] >= 6, 'quality'] = 1

input_cols=list(df.columns)[:-1]
output_cols = ['quality']
print(input_cols, output_cols)

#Preparem dataset per al training:

def dataframe_to_arrays(dataframe):
    # Copiem df original
    dataframe1 = df.copy(deep=True)

    # Convertim columnes categòriques en numériques (en aquest cas no cal!!)
    #for col in categorical_cols:
        #dataframe1[col] = dataframe1[col].astype('category').cat.codes

    # Convertim a numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(df)
print(inputs_array, targets_array)


#Ara convertim els numpy arrays a tensors

inputs = torch.from_numpy(inputs_array).type(torch.float)
targets = torch.from_numpy(targets_array).type(torch.float)

print('Shape of input tensor and target tensor::  ', inputs.shape, targets.shape)
print('datatype of input tensor and target tensor::  ', inputs.dtype, targets.dtype)

dataset = TensorDataset(inputs, targets)


#Separem dades train i test

train_ds, val_ds = random_split(dataset, [1300, 299]) #Divideix aleatòriament un conjunt de dades en nous conjunts de dades no superposats
batch_size=50 #num de mostres abans d'actualitzar els paràmetres interns del model
train_loader = DataLoader(train_ds, batch_size, shuffle=True) #Produeix conjunts de dades per cada epoch, shuffle=T barreja les dades
val_loader = DataLoader(val_ds, batch_size)#no cal barrejar, només és per validar

#Creem model de regressió lineal

input_size = len(input_cols)
output_size = len(output_cols)


class WineQuality(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,
                                output_size)  # Crea paràmetres del model?

    def forward(self, xb):
        out = self.linear(xb)  # Com es fa forward amb un model lineal?
        return out

    def training_step(self, batch):
        inputs, targets = batch
        # Genera prediccions
        out = self(inputs)
        # Calcula la loss
        loss = F.l1_loss(out, targets)  # compara prediccions amb realitat
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(out, targets)  # compara prediccions amb realitat per validació final
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combina losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Print cada 100 epoch
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result['val_loss']))


model=WineQuality()


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


#Entrenem el model:

epochs = 1500
lr = 1e-6
history5 = fit(epochs, lr, model, train_loader, val_loader)


def get_accuracy(pred_arr,original_arr):
    pred_arr = pred_arr.detach().numpy()
    original_arr = original_arr.numpy()
    final_pred = []

#Fem prediccions amb el model resultant:

def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)


input, target = val_ds[62]
predict_single(input, target, model)

test_acc = get_accuracy(output_test, y_test)