"""
problem tanimi: lstm ile metin turetme
"""

# lstm hatirlatma

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter # kelime frekanslarini hesaplamak icin 
from itertools import product # grid search icin kombinasyon olusturmak

# %% veri yukleme ve on isleme (preprocessing)
# urun yorumlari
text = """Bu ürün beklentimi fazlasıyla karşıladı.  
Malzeme kalitesi gerçekten çok iyi.  
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.  
Fiyatına göre performansı harika.  
Kesinlikle tavsiye ederim ve öneririm!"""

# veri on ısleme: 
# noktalama isaretlerinden kurtul, 
# kucuk harf donusumu
# kelimeleri bol

words = text.replace(".", "").replace("!","").lower().split()

# kelime frekanslarini hesapla ve indeksleme olustur
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse = True) # kelime frekansini buyukten kucuge sirala
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# egitim verisi hazirlama
data = [(words[i], words[i+1]) for i in range(len(words) - 1)]

# %% lstm modeli tanimlama

class LSTM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        
        super(LSTM, self).__init__() # bir ust sinifin constructor ini cagirma
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # embedding katmani
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) # LSTM katmani
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x): # ileri besleme fonksiyonu
        """
            input -> embedding -> lstm -> fc -> output
        """
        
        x = self.embedding(x) # input -> embedding
        lstm_out, _ = self.lstm(x.view(1,1,-1))
        output = self.fc(lstm_out.view(1,-1))
        return output 

model = LSTM(len(vocab), embedding_dim=8, hidden_dim=32)

# %% hyperparameter tuning 

# kelime listesi -> tensor
def prepare_squence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype = torch.long)

# hyperparameter tuning kombinasyonlarini belirle
embedding_sizes = [8, 16] # denenecek embedding boyutlari
hidden_sizes = [32, 64] # denenecek gizli katman boyutlari
learning_rates = [0.01, 0.005] # ogrenme orani

best_loss = float("inf") # en dusuk kayip degerini saklamak icin bir degisken
best_params = {} # en iyi parametreleri saklamak icin bos bir dictionary

print("Hyperparameter tuning basliyor ...")

# grid search
for emb_size, hidden_size, lr in product(embedding_sizes, hidden_sizes, learning_rates):
    print(f"Deneme: Embedding: {emb_size}, Hidden: {hidden_size}, learning_rate: {lr}")
    
    # modeli tanimla
    model = LSTM(len(vocab), emb_size, hidden_size) # secilen parametreler ile model olustur
    loss_function = nn.CrossEntropyLoss() # entropi kayip fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr = lr) # secilen lr ile adam optimizeri

    epochs = 50
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0 # epoch baslangicinda kaybi sifirlayalim
        for word, next_word in data: 
            model.zero_grad() # gradyanlari sifirla
            input_tensor = prepare_squence([word], word_to_ix) # girdiyi tensor e cevir
            target_tensor = prepare_squence([next_word], word_to_ix) # hedef kelimeyi tensore donustur
            output = model(input_tensor) # prediction
            loss = loss_function(output, target_tensor)
            loss.backward() # geri yayilim islemi uygula
            optimizer.step() # parametreleri guncelle
            epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.5f}")
        total_loss = epoch_loss 
    
    # en iyi modeli kaydet
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_dim": emb_size, "hidden_dim":hidden_size, "learning_rate":lr}
    print()

print(f"Best params: {best_params}")

# %% lstm training

final_model = LSTM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'])
optimizer = optim.Adam(final_model.parameters(), lr = best_params['learning_rate'])
loss_function = nn.CrossEntropyLoss() # entropi kayip fonksiyonu

print("Final model training")
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0 
    for word, next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_squence([word], word_to_ix)
        target_tensor = prepare_squence([next_word], word_to_ix)
        output = final_model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Final Model Epoch: {epoch}, Loss: {epoch_loss:.5f}")


# %% test ve degerlendirme

# kelime tahmini fonksiyonu: baslangic kelimesi ve n adet kelime uretmesini sagla
def predict_sequence(start_word, num_words):
    current_word = start_word # suanki kelime baslangic kelimesi olarak ayarlanir
    output_sequence = [current_word] # cikti dizisi 
    
    for _ in range(num_words): # belirtilen sayida kelime tahmini
        with torch.no_grad(): # gradyan hesaplamasi yapmadan
            input_tensor = prepare_squence([current_word], word_to_ix) # kelime -> tensor
            output = final_model(input_tensor)
            predicted_idx = torch.argmax(output).item() # en yuksek olasiliga sahip kelimenin indexi
            predicted_word = ix_to_word[predicted_idx] # indekse karsilik gelen kelimeyi return eder
            output_sequence.append(predicted_word)
            current_word = predicted_word # bir sonraki tahmin icin mevcut kelimeleri guncelle
    return output_sequence # tahmin edilen kelime dizisi return edilir.
       
"""
Bu ürün beklentimi fazlasıyla karşıladı.  
Malzeme kalitesi gerçekten çok iyi.  
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.  
Fiyatına göre performansı harika.  
Kesinlikle tavsiye ederim ve öneririm!
"""     
start_word = "ve"           
num_predictions = 2
predicted_sequence = predict_sequence(start_word, num_predictions)
print(" ".join(predicted_sequence))

















