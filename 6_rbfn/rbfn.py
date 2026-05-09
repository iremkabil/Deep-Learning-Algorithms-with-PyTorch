# https://archive.ics.uci.edu/dataset/53/iris

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %% veri setinin iceriye aktarilmasi
# siniflandirma problemi: iris veri seti 3 farkli sinifa ait siniflandirma problemi

df = pd.read_csv("iris.data", header = None)

X = df.iloc[:, :-1].values # ilk 4 sutunu x degiskenine atar
y, _ = pd.factorize(df.iloc[:, -1])

# veriyi standardize et
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train test split olarak ikiye ayir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

def to_tensor(data, target):
    return torch.tensor(data, dtype = torch.float32), torch.tensor(target, dtype = torch.long)

X_train, y_train = to_tensor(X_train, y_train)
X_test, y_test = to_tensor(X_test, y_test)


# %% RBFN modelinin ve rbf_kernel'in tanimlanmasi

def rbf_kernel(X, centers, beta):
    return torch.exp(-beta * torch.cdist(X, centers)**2)

class RBFN(nn.Module):
    
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim)) # rbf merkezlerini rastgele baslat
        self.beta = nn.Parameter(torch.ones(1) * 2.0) # beta parametresi rbf in genisligini kontrol edece
        self.linear = nn.Linear(num_centers, output_dim) # outputu tam baglantili katmana yonlendir
    
    def forward(self, x): # ileri yayilim
        # rbf cekirdek fonksiyonunu hesapla
        phi = rbf_kernel(x, self.centers, self.beta)
        return self.linear(phi)

# model = RBFN(4, 10, 3)
# %% model training

num_centers = 10
model = RBFN(input_dim=4, num_centers=num_centers, output_dim=3)

# kayip fonksyionu tanimlama ve optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# modeli egitelim
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad() # gradyanlari sifirla
    outputs = model(X_train) # prediction yani iler yayilim
    loss = criterion(outputs, y_train) # loss hesapla
    loss.backward() # geri yayilim
    optimizer.step() # agirliklari guncelle
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


# %% test and evaluation

with torch.no_grad():
    y_pred = model(X_test) # test verisi ile tahmin et
    accuracy = (torch.argmax(y_pred, axis=1) == y_test).float().mean().item() # dogruluk hesapla
    print(f"accuracy: {accuracy}")













