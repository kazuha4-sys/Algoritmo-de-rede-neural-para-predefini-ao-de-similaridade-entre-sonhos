import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

class DreamNetwork(nn.Module):
    def __init__(self, input_dim):
        super(DreamNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def preprocess_text(dream):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(dream.lower())
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

def train_model(X, y):
    input_dim = X.shape[1]
    model = DreamNetwork(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    return model

def predict_similarity(model, vectorizer, dream1, dream2):
    dream1_processed = preprocess_text(dream1)
    dream2_processed = preprocess_text(dream2)
    
    dreams = [dream1_processed, dream2_processed]
    X = vectorizer.transform(dreams).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(X_tensor[0] - X_tensor[1])
    
    return output.item()

# Exemplo de uso
dreams = [
    "I was flying over mountains and rivers",
    "I was swimming in the ocean with dolphins",
    "I was falling from the sky"
]

relations = [0, 1, 0]  # 1 significa que os sonhos são relacionados, 0 significa que não são

vectorizer = TfidfVectorizer()
dreams_processed = [preprocess_text(dream) for dream in dreams]
X = vectorizer.fit_transform(dreams_processed).toarray()

model = train_model(X, np.array(relations))

dream1 = "I was flying in the sky"
dream2 = "I was falling from a tall building"

similarity = predict_similarity(model, vectorizer, dream1, dream2)
print(f"Similaridade entre os sonhos: {similarity:.4f}")
