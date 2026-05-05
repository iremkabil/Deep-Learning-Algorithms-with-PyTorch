# %% library
import torch      # tensor islemleri
import torch.nn as nn  ## yapay sinir agi katmanlarini tanimlamak icin kullan
import torch.optim as optim  # optimizasyon algortimalarini iceren modul
import torchvision           # goruntu isleme ve pre-defined modelleri icerir
import torchvision.transforms as transforms  # goruntu donusumleri yapmak
import matplotlib.pyplot as plt    

# optional
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading

def get_data_loaders(batch_size = 64): #her iterasyonda islenecek veir miktari
    
    transform = transforms.Compose([
        transforms.ToTensor(), # goruntuyu tensore cevirir ve 0-255 -> 0-1 olceklendir.
        transforms.Normalize((0.5,), (0.5,)) # piksel degerlerini -1 ile 1 arasina olcekler
        ])
    
    train_set = torchvision.datasets.MNIST(root = "./data", train = True, download =True, transform = transform)
    test_set = torchvision.datasets.MNIST(root = "./data", train = True, download =True, transform = transform)
    
    # pytorch veri yukleyicisini olustur
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
    
# train_loader, test_loader = get_data_loaders()

# data visualization
def visualize_samples(loader, n):
    images, labels = next(iter(loader)) # ilk batch' ten goruntu ve etiketleri alalim
    fig, axes = plt.subplots(1, 5, figsize=(10,5)) # n farkli goruntu icin gorsellestirme alani
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap = "gray")
        axes[i].set_title(f"Label: {labels[i].item()}")  # goruntuye ait sinif etiketini baslik olarak yaz
        axes[i].axis("off")  # eksenleri gizli
        
    plt.show()

# visualize_samples(train_loader, 4)
    
# %% define ann model
    
# yapay sinir agi class
class NeuralNetwork(nn.Module):  # pytorch'un nn.Module sinifindan miras aliyor
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
    
        # elimizde bulunan goruntuleri vektor haline cevirelim (1D)
        self.flatten = nn.Flatten()
        
        # ilk tam bagli katmani olustur
        self.fc1 = nn.Linear(28*28, 128)   # input size=784 , output size = 128
        
        # aktivasyon fonksiyonu olustur
        self.relu = nn.ReLU() 
        
        # ikinci tam bagli katmani olustur
        self.fc2 = nn.Linear(128, 64)
        
        # cikti katmani olustur
        self.fc3 = nn.Linear(64, 10) # cikti katmani olustur: 
        
    def forward(self, x):  # forward propogation: ileri yayilim, giris olarak x goruntu alsin
    
        x = self.flatten(x) # initial x = 28*28 lik bir goruntu -> duzlestir 784 vektor haline getir
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  

        return x # modelimizin ciktsini return edelim
    
#create model and compile
# model = NeuralNetwork().to(device)

# kayip fonk. ve optimizasyon algoritmasini belirle
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),          # multi class classification problems loss function
    optim.Adam(model.parameters(), lr = 0.0001)  # update weights with adam
)

# criterion, optimizer = define_loss_and_optimizer(model)

# %% train
def train_model(model, train_loader, criterion, optimizer, epochs = 10):
    model.train()                             # modelimizi egitim moduna alalim
    
    train_losses = [] # her bir epoch sonucunda elde edilen loss degerlerini saklamak icin bir liste
    
    for epoch in range(epochs): # belirtilen epoch sayisi kadar egitim yapalim
        total_loss = 0
        
        for images, labels in train_loader: # tum eegitim verileri uzerinde iterasyon gerceklestir
            images, labels = images.to(device), labels.to(device)# verileri cihaza tasi
    
            optimizer.zero_grad()          # gradyanlari sifirla 
            predictions = model(images)    # modeli uygula, forward pro.
            loss = criterion(predictions, labels) # loss hesaplama -> y_prediction ile y_real
            loss.backward()                # geri yayilim yani gradyan hesaplama
            optimizer.step()               # update weights
            
            total_loss = total_loss + loss.item()
            
        avg_loss = total_loss / len(train_loader) # ortalama kayip hesaplama
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} / {epochs}, Loss:{avg_loss:.3f} ")
        
    # loss graph
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

# train_model(model, train_loader, criterion, optimizer, epochs=1)
# %% test
def test_model(model, test_loader):
    model.eval()  #modelimizi degerlendirme moduna al
    correct = 0     # dogur tahmin sayaci
    total = 0       # toplam veri sayisi
    
    with torch.no_grad():   # gradyan hesaplama gereksiz
        for images, labels in test_loader: #test veri kumesini donguye al
            images,labels = images.to(device), labels.to(device) # verileri cihaza tasi
            predictions = model(images)
            _, predicted = torch.max(predictions, 1)  # en yuksek olasilik sinifin etiketini bul
            total += labels.size(0)  # toplam veri sayisini guncelle
            correct += (predicted == labels).sum().item()  #dogru tahminleri say
            
    print(f"Test Accuracy: {100*correct/total:.3f}%")
# test_model(model, test_loader)
            

# %% main

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()  #veri yukleyicilerini al
    visualize_samples(train_loader, 5)
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)









