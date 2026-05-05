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
    
train_loader, test_loader = get_data_loaders()

# data visualization
def visualize_samples(loader, n):
    images, labels = next(iter(loader)) # ilk batch' ten goruntu ve etiketleri alalim
    fig, axes = plt.subplots(1, 5, figsize=(10,5)) # n farkli goruntu icin gorsellestirme alani
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap = "gray")
        axes[i].set_title(f"Label: {labels[i].item()}")  # goruntuye ait sinif etiketini baslik olarak yaz
        axes[i].axis("off")  # eksenleri gizli
        
    plt.show()

visualize_samples(train_loader, 4)
    
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







# %% train















