import torch  # PyTorch kütüphanesini içe aktarıyoruz
import torch.nn as nn  # Sinir ağı katmanları için PyTorch modülünü içe aktarıyoruz
import torch.optim as optim  # Optimizasyon algoritmalarını içe aktarıyoruz
import torchvision.transforms as transforms  # Görüntü dönüşümleri için modülü içe aktarıyoruz
import torchvision.datasets as datasets  # Hazır veri setlerini kullanmak için modülü içe aktarıyoruz
import torchvision.models as models  # Önceden eğitilmiş modelleri yüklemek için modülü içe aktarıyoruz
from torch.utils.data import DataLoader  # Veri yükleyici için PyTorch modülünü içe aktarıyoruz
from tqdm import tqdm  # Eğitim sürecini izlemek için ilerleme çubuğunu içe aktarıyoruz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Cihaz seçimi (GPU varsa kullan, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri seti dönüşümleri (Data Augmentation eklendi)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNet giriş boyutunu 224x224 olarak ayarlıyoruz
    transforms.RandomHorizontalFlip(),  # Görüntüleri yatay çevirerek veri artırımı yapıyoruz
    transforms.RandomRotation(10),  # Görüntüleri rastgele 10 dereceye kadar döndürüyoruz
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Renk varyasyonları ekliyoruz
    transforms.ToTensor(),  # Görüntüleri tensöre çeviriyoruz
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Piksel değerlerini normalize ediyoruz
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Test veri seti için giriş boyutunu ayarlıyoruz
    transforms.ToTensor(),  # Görüntüleri tensöre çeviriyoruz
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Piksel değerlerini normalize ediyoruz
])

# Oxford Flowers 102 veri setini yükleme (Eğitim ve test veri setleri)
train_dataset = datasets.Flowers102(root="./data", split="train", transform=transform_train, download=True)  # Eğitim veri setini indiriyoruz
test_dataset = datasets.Flowers102(root="./data", split="val", transform=transform_test, download=True)  # Test veri setini indiriyoruz

# Rastgele 5 örnek seçme
indices = torch.randint(len(train_dataset), (5,))
samples = [train_dataset[i] for i in indices]

# Görselleştirme
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, (image, label) in enumerate(samples):
    image = image.numpy().transpose((1, 2, 0))  # Tensörü görüntü formatına dönüştürme
    image = (image * 0.5) + 0.5  # Normalizasyonu geri alma
    axes[i].imshow(image)
    axes[i].set_title(f"Class: {label}")
    axes[i].axis("off")
plt.show()

# Veri yükleyicileri oluşturma
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Eğitim veri yükleyicisini oluşturuyoruz
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # Test veri yükleyicisini oluşturuyoruz

# %% 

# MobileNetV2 modelini yükleme (Önceden eğitilmiş ağırlıklarla)
model = models.mobilenet_v2(pretrained=True)  # MobileNetV2 modelini indiriyoruz

# Sınıflandırıcı katmanı değiştirme (Çıkış sayısını 102 sınıfa uyarlıyoruz)
num_ftrs = model.classifier[1].in_features  # Mevcut sınıflandırıcı katmanının giriş özellik sayısını alıyoruz
model.classifier[1] = nn.Linear(num_ftrs, 102)  # Son katmanı Oxford Flowers 102 için değiştiriyoruz
model = model.to(device)  # Modeli belirtilen cihaza (GPU/CPU) taşıyoruz

# Kayıp fonksiyonu ve optimizer tanımlama
criterion = nn.CrossEntropyLoss()  # Çok sınıflı sınıflandırma için çapraz entropi kaybı kullanıyoruz
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)  # Adam optimizasyon algoritmasını kullanıyoruz
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # StepLR ekleme

# Modeli eğitme işlemi
epochs = 3  # Eğitim döngüsünü 25 epoch olarak belirliyoruz
for epoch in tqdm(range(epochs)):  # Her epoch için ilerleme çubuğu gösteriyoruz
    model.train()  # Modeli eğitim moduna alıyoruz
    running_loss = 0.0  # Kümülatif kayıp değerini sıfırlıyoruz
    for images, labels in tqdm(train_loader):  # Eğitim veri yükleyicisini döngüye alıyoruz
        images, labels = images.to(device), labels.to(device)  # Görüntüleri ve etiketleri cihaza taşıyoruz
        optimizer.zero_grad()  # Önceki gradyanları sıfırlıyoruz
        outputs = model(images)  # Model ile tahmin yapıyoruz
        loss = criterion(outputs, labels)  # Kayıp değerini hesaplıyoruz
        loss.backward()  # Geri yayılım (backpropagation) işlemini gerçekleştiriyoruz
        optimizer.step()  # Ağırlıkları güncelliyoruz
        running_loss += loss.item()  # Toplam kayıp değerini güncelliyoruz
    scheduler.step()  # Öğrenme oranı güncelleme
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")  # Epoch sonunda ortalama kaybı yazdırıyoruz
    
# modeli kaydetme
torch.save(model.state_dict(), "mobilenet_flowers102.pth")
# %% test and evaluting
    
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
# confusion matrix
cm = confusion_matrix(all_labels, all_preds) 
plt.figure(figsize = (12,12))
sns.heatmap(cm, annot = False, cmap = "Blues")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.show()   

print(classification_report(all_labels, all_preds))    
    
