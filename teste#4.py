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

# Definição da Rede Neural
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

# Pré-processamento de texto
def preprocess_text(dream):
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(dream.lower())
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# Função para treinar o modelo
def train_model(X, y):
    input_dim = X.shape[1]
    model = DreamNetwork(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Verificação do tamanho dos tensores
    if X_tensor.size(0) != y_tensor.size(0):
        raise ValueError(f"Tamanhos incompatíveis: X_tensor tem {X_tensor.size(0)} amostras e y_tensor tem {y_tensor.size(0)} amostras.")
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()  # Zera os gradientes do otimizador
        outputs = model(X_tensor)
        
        # Verificação do tamanho da saída
        if outputs.size() != y_tensor.size():
            raise ValueError(f"Tamanhos incompatíveis: outputs tem {outputs.size()} e y_tensor tem {y_tensor.size()}.")
        
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    return model

# Função para prever a similaridade entre dois sonhos
def predict_similarity(model, vectorizer, dream1, dream2):
    dream1_processed = preprocess_text(dream1)
    dream2_processed = preprocess_text(dream2)
    
    dreams = [dream1_processed, dream2_processed]
    X = vectorizer.transform(dreams).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Autograd está ativa aqui, então os gradientes serão calculados
    output = model(X_tensor[0] - X_tensor[1])
    
    return output.item()

# Função para carregar sonhos de um arquivo .txt
def load_dreams_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        dreams = [line.strip() for line in file.readlines()]
    return dreams

# Exemplo de uso
dreams = load_dreams_from_file('sonhos.txt')  # Carrega os sonhos do arquivo 'sonhos.txt'
relations = [0, 1, 0, 1]  # Exemplo de relações entre os sonhos (ajuste conforme necessário)

# Verifica se o número de relações corresponde ao número de sonhos
if len(dreams) != len(relations):
    raise ValueError(f"O número de sonhos ({len(dreams)}) deve corresponder ao número de relações ({len(relations)}).")

vectorizer = TfidfVectorizer()
dreams_processed = [preprocess_text(dream) for dream in dreams]
X = vectorizer.fit_transform(dreams_processed).toarray()

model = train_model(X, np.array(relations))

dream1 = "Eu estava voando no céu"
dream2 = "Eu estava caindo de um prédio alto"

similarity = predict_similarity(model, vectorizer, dream1, dream2)
print(f"Similaridade entre os sonhos: {similarity:.4f}")
