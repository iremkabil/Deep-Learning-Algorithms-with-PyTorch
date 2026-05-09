"""
siniflandirma projesi gerceklestir
positive ve negative commentlerden olusan bir veri seti

"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split # veri setini trainve test olmak uzere ikiye boler
from sklearn.metrics import accuracy_score
import string
from collections import Counter

# %% verinin tanimlanmasi ve veri on isleme

positive_sentences = [
    "This is amazing!", "I love this!", "Absolutely fantastic!", "Highly recommended!", "Best experience ever!",
    "Superb quality!", "Exceeded my expectations!", "A wonderful experience!", "Totally worth it!", "Would buy again!",
    "Five stars!", "Incredible product!", "I'm very happy with this!", "Outstanding!", "Brilliant service!",
    "So much fun!", "Excellent craftsmanship!", "Really impressive!", "Made my day!", "Couldn't be happier!",
    "Top-notch quality!", "Fantastic design!", "Love the details!", "Highly satisfying!", "Absolutely love it!",
    "Couldn't ask for more!", "Very well made!", "A game-changer!", "Best purchase ever!", "I'm in love with this!",
    "Simply the best!", "Perfect in every way!", "Extremely useful!", "A must-have!", "Worth every penny!",
    "The quality is outstanding!", "Unmatched excellence!", "So convenient!", "Pleasantly surprised!",
    "Great attention to detail!", "High quality and durable!", "Delightful experience!", "Brings joy every time!",
    "I appreciate this so much!", "Super comfortable!", "Amazing customer service!", "Couldn't be better!",
    "Such a great find!", "Really well designed!", "Love the functionality!", "Great job!", "Impressed beyond words!",
    "Super stylish!", "Better than expected!", "Everything I wanted!", "A true masterpiece!", "Simply wonderful!",
    "Great investment!", "The best choice I've made!", "High-end quality!", "A truly remarkable item!",
    "Will definitely recommend!", "Brilliantly made!", "Love the colors!", "So much value for the price!",
    "Makes life easier!", "One of a kind!", "I'm thrilled!", "So elegant!", "Perfectly designed!",
    "A real pleasure to use!", "Great for everyday use!", "An absolute joy!", "Beyond happy with this!",
    "Couldn't have asked for more!", "Definitely worth it!", "Such a smart design!", "Wonderful packaging!",
    "So practical!", "A real lifesaver!", "Super functional!", "Top-tier experience!", "An instant favorite!",
    "Really stands out!", "Best in its class!", "Excellent durability!", "A pleasure to have!",
    "The craftsmanship is superb!", "Exceeds all expectations!", "Simply outstanding!", "Just what I needed!",
    "I’m totally satisfied!", "Great for gifting!", "Absolutely flawless!", "Couldn't be any better!",
    "A work of art!", "A real gem!", "One of the best things I've bought!", "Super well made!",
    "A fantastic surprise!", "Genuinely impressive!", "This made me so happy!", "It's just perfect!",
    "Couldn't be more pleased!", "A fantastic addition!", "So refreshing!", "Brightens up my day!","I love this product.", "This is amazing!", "I had a great experience.", "This is fantastic.",
    "I am so happy with this.", "I would recommend this to everyone.", "Absolutely wonderful experience!",
    "This is perfect!", "Such a great product!", "I can't get enough of this.", 
    "Very satisfied with my purchase.", "Highly recommended!", "It exceeded my expectations.", 
    "This is exactly what I needed.", "Great value for money.", "I would buy this again.", 
    "I am very pleased with it.", "It works just as described.", "Such a useful product!", 
    "I’m really impressed with the quality.", "The best purchase I’ve made.", "It makes my life easier.",
    "It was a great decision to buy this.", "This is exactly what I was looking for.", 
    "Amazing quality and performance.", "I can’t recommend this enough.", "I’m so glad I found this.",
    "Such a convenient product.", "It does exactly what I need.", "Fantastic quality!", "This is a must-have product.",
    "So easy to use!", "It works perfectly every time.", "Incredible design.", "Very high quality product.", 
    "I am totally satisfied.", "Amazing customer service.", "I’m never going back to the old way.",
    "Top-notch performance.", "Superb quality.", "This product is life-changing.", "Highly effective.", 
    "Best decision ever!", "It’s been so helpful.", "Perfect for everyday use.", "I’ll be using this for a long time.", 
    "Really makes a difference.", "I recommend this to everyone.", "Great addition to my collection.",
    "I am loving this product.", "Such good quality.", "I’m so impressed with this.", "It’s made my tasks easier.",
    "This is an outstanding product.", "It’s made my life better.", "Incredible product.", "Wonderful investment.",
    "I love everything about this.", "Such a great purchase.", "Best purchase I’ve ever made.", 
    "I’m a huge fan of this product.", "This is exactly what I wanted.", "I will always buy this.", "Excellent performance.",
    "I’m so happy I chose this.", "It’s been an amazing experience.", "I can’t imagine living without it.", 
    "I’m extremely happy with this.", "Great quality for the price.", "I’ve never been this satisfied.", 
    "I trust this product completely.", "I use this every day.", "It works seamlessly.", "It has made a huge difference.",
    "Perfect for what I need.", "I’m beyond happy with this.", "This product works wonders.", "This is my favorite product.",
    "I will never regret this purchase.", "A game changer for me.", "Couldn’t be more satisfied.", "I will always recommend this.",
    "This product is amazing.", "Completely worth it.", "Very reliable.", "I’m very happy with the results.", 
    "This product has exceeded my expectations.", "Impressive quality.", "This product does not disappoint.",
    "I’m very grateful for this product.", "Truly incredible.", "Best in its class.", "You won’t regret buying this.",
    "I’m extremely pleased with this.", "Everything about this product is perfect.", "I would definitely buy it again.",
    "This product makes a real difference.", "I love how easy it is to use.", "It’s everything I expected and more.",
    "One of the best products I’ve ever bought.", "I’ve been using it nonstop.", "This is everything I wanted.",
    "I can’t imagine life without this product.", "So glad I purchased this.", "You can’t go wrong with this.", 
    "A fantastic product.", "This is the best purchase I’ve made in a while.", "I’m in love with this product.",
    "So reliable.", "So happy I discovered this.", "I feel like this product was made for me.", "Everything works perfectly.",
    "I’m thoroughly satisfied.", "I would recommend this to anyone.", "So simple and easy to use.", "Top quality.", 
    "I’m so impressed.", "This product really does work.", "An excellent choice.", "I’ve never been this happy with a product.",
    "This is by far the best product in this category.", "A must-have for anyone.", "Couldn’t ask for more.", "Absolutely perfect.",
    "I am so happy with my purchase.", "I’m completely satisfied with this.", "This has made my life so much easier.",
    "Definitely worth the price.", "I’m so glad I bought this.", "Super high quality.", "Such a great buy.",
    "One of my favorite purchases."
]

negative_sentences = [
    "I do not like this.", "This is terrible!", "Very disappointing.", "Not what I expected.", "Worst experience ever.",
    "Extremely poor quality.", "Would not recommend.", "Absolutely awful!", "A complete waste of money.", "So frustrating!",
    "One star.", "Horrible product!", "I'm very unhappy with this.", "Really bad.", "Terrible service!",
    "This is so annoying.", "Extremely poorly made.", "Very unimpressive.", "Ruined my day!", "Couldn’t be worse.",
    "Total disaster!", "Horrible design.", "The details are lacking.", "Highly unsatisfying.", "I absolutely hate it.",
    "Wish I never bought this.", "So poorly made!", "A huge letdown.", "Worst purchase ever.", "This is just bad.",
    "Not worth the price.", "Completely useless.", "A total ripoff!", "Very cheap quality.", "So disappointed.",
    "The quality is terrible.", "Extremely frustrating.", "Absolutely useless.", "Very inconvenient.",
    "Lack of attention to detail.", "Feels flimsy and cheap.", "Horrible experience!", "Brings me nothing but stress.",
    "I regret this so much.", "Very uncomfortable.", "Terrible customer service.", "This is a mess.",
    "Such a terrible find!", "Really badly designed.", "The functionality is awful.", "This is just wrong.",
    "So ugly!", "Much worse than expected.", "Not what I wanted at all!", "A total failure!", "Simply awful!",
    "A terrible investment!", "The worst choice I've made!", "Cheap and flimsy.", "A truly awful item!",
    "Never buying this again!", "Poorly manufactured.", "The colors are so bad!", "Not worth a single penny.",
    "Makes life harder!", "Nothing special at all.", "I'm beyond disappointed!", "So tacky!", "Completely ruined!",
    "Awfully designed!", "Not practical at all.", "A real pain to use.", "Beyond frustrating!", "Such a letdown!",
    "Definitely not worth it!", "Very poorly thought out.", "Terrible packaging!", "So impractical!",
    "A real nightmare!", "Super dysfunctional.", "Bottom-tier experience.", "Will never use this again.",
    "Really terrible!", "Worst in its category.", "Falls apart so easily!", "A horrible mistake.",
    "The craftsmanship is awful!", "Fails to meet expectations.", "Simply unacceptable!", "Not what I needed at all.",
    "I’m totally dissatisfied!", "Horrible for gifting!", "Absolutely defective!", "Couldn't be any worse!",
    "An absolute failure!", "A real eyesore!", "One of the worst things I've bought!", "So poorly constructed!",
    "A frustrating surprise!", "Genuinely disappointing!", "This ruined my mood!", "It's just awful!",
    "Couldn't be more disappointed!", "A frustrating addition!", "So unreliable!", "Brings me nothing but problems!", "I hate this product.", "This is terrible.", "I had a bad experience.", "This is awful.", "I am not happy with this.",
    "I would never recommend this.", "This is a disaster.", "Totally useless.", "I regret buying this.", "Worst product ever.",
    "This is not what I expected.", "Not satisfied with this.", "The quality is poor.", "This didn’t work as advertised.",
    "Very disappointing.", "I’m unhappy with my purchase.", "I would never buy this again.", "This product is garbage.",
    "Not worth the money.", "It broke within a week.", "The product doesn’t work.", "Completely useless.", "I’m frustrated with this product.",
    "This product didn’t meet my expectations.", "Very poor quality.", "The customer service was awful.", "Terrible performance.",
    "This was a mistake.", "Not at all impressed.", "I’m so disappointed with this.", "This didn’t live up to the hype.",
    "It didn’t do what I needed.", "The product was defective.", "I’m dissatisfied with my purchase.", "The quality is subpar.",
    "It’s not as described.", "I will never buy this again.", "I don’t like this product at all.", "A huge waste of money.",
    "I really regret buying this.", "This is one of the worst products I’ve bought.", "The product stopped working quickly.",
    "Not recommended at all.", "This didn’t solve my problem.", "It’s completely ineffective.", "I’m upset with this product.",
    "It’s not worth the hype.", "This is a rip-off.", "Very disappointed in this purchase.", "The product was faulty.",
    "This doesn’t work at all.", "It broke too easily.", "It doesn’t live up to the expectations.", "I feel cheated.",
    "It didn’t even last a day.", "The quality is terrible.", "Waste of money.", "I will never buy anything from this brand again.",
    "This was a total letdown.", "I don’t recommend this to anyone.", "Very unreliable.", "Not good quality.", "It was a huge disappointment.",
    "Not as advertised.", "I’m extremely unhappy with this product.", "This product didn’t perform well.", "It’s worse than I expected.",
    "It’s a scam.", "This product is awful.", "I regret purchasing this.", "Very low quality.", "This is such a poor product.",
    "It’s completely useless.", "This is a rip-off.", "I can’t believe I bought this.", "The product doesn’t work as intended.",
    "A waste of time and money.", "Extremely disappointed.", "This product doesn’t work at all.", "This was a bad decision.",
    "This product is not worth it.", "Horrible quality.", "I’ll never use this again.", "I’ll return this as soon as I can.",
    "Such a disappointment.", "The product stopped working after one use.", "I don’t recommend this at all.", 
    "This is by far the worst product I’ve ever bought.", "The worst purchase ever.", "This is not reliable.", 
    "The product is defective.", "I’m really unhappy with this.", "So disappointed with my purchase.", "Not happy at all.",
    "Very poor value for money.", "I wouldn’t recommend this to anyone.", "It broke on the first use.", "It didn’t meet my needs.",
    "Not worth buying.", "I can’t believe I wasted my money on this.", "I regret spending my money on this.", "Such a bad experience.",
    "The product is useless.", "Terrible experience.", "I can’t believe this is so bad.", "The product arrived broken.",
    "I am unhappy with the quality.", "The product didn’t even work for a day.", "I wish I never bought this.",
    "It didn’t do anything.", "I’m thoroughly disappointed.", "Very low performance.", "This product is a joke.",
    "The performance is terrible.", "This is a total failure.", "This product is nothing like what was advertised.",
    "I feel like I’ve been ripped off.", "This is not worth the price at all."
]

# veri on isleme
def preprocess(text):
    text = text.lower() # tum metni kucuk harfe cevir
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text 

# veri seti olusturma
data = positive_sentences + negative_sentences
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)

# veriyi on isle
data = [preprocess(sentence) for sentence in data]

# vocab olsutur (kelime dagarcigi)
all_words = " ".join(data).split()
word_counts = Counter(all_words)
vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0 # padding ozel token tanimla

# veriyi tensorler cevirme
max_len = 15
def sentence_to_tensor(sentence, vocab, max_len = 15):
    tokens = sentence.split() # cumleyi tokenlara yani kelimelere ayir
    indices = [vocab.get(word,0) for word in tokens] # indeksleri alma
    indeces = indices[:max_len]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices)

X = torch.stack([sentence_to_tensor(sentence, vocab, max_len) for sentence in data])
y = torch.tensor(labels)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# %% transformer modelinin olusturulmasi
class TransformerClass(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes):
        
        super(TransformerClass, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        self.transformer = nn.Transformer(d_model = embedding_dim, # embedding vektor boyut
                                          nhead = num_heads, # multi head attention mekanizmasindaki baslik sayisi
                                          num_encoder_layers = num_layers, # transformer encode katmani sayisi
                                          dim_feedforward=hidden_dim) # encoder icinde bulunan gizli katman boyutu
        
        self.fc = nn.Linear(embedding_dim*max_len, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding
        output = self.transformer(embedded, embedded)
        output = output.view(output.size(0), -1)
        output = torch.relu(self.fc(output))
        output = self.out(output)
        output = self.sigmoid(output)
        return output

# model = TransformerClass(len(vocab), 32, 4, 4, 64, 1)
     
# %% tranining

vocab_size = len(vocab)
embedding_dim = 32
num_heads = 4
num_layers = 4
hidden_dim = 64
num_classes = 1 # olumlu yorumlar ve olumsuz yorumlar

model = TransformerClass(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes)

# loos ve optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

# training
number_epochs = 30
model.train() # modeli training moduna al
for epoch in range(number_epochs):
    optimizer.zero_grad() # gradyanlari sifirla
    output = model(X_train.long()).squeeze()
    loss = criterion(output, y_train.float())
    loss.backward() # gradyanlari hesapla
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{number_epochs} Loss: {loss}")

# %% test 
model.eval()
with torch.no_grad():
    y_pred = model(X_test.long()).squeeze()
    y_pred = (y_pred > 0.5).float()
    
    y_pred_training = model(X_train.long()).squeeze()
    y_pred_training = (y_pred_training > 0.5).float()
    
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")

accuracy_train = accuracy_score(y_train, y_pred_training)
print(f"Train accuracy: {accuracy_train}")






















