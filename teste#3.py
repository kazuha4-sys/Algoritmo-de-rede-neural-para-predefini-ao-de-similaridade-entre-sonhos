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
            x = torch.sifmoid(self.fc3(x))
            return x 
        
def preprocess_text(dream):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(dream.lower())
    # Filtrar as plavars, removendo stopwords e mantendo apenas palavras alfanúmerico
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words) # Rotorna o texto filtrado comm uma unica string
 
# Treinamento do modelo 
def train_model(X, y):
    input_dim = X.shape[1]
    model = DreamNetwork(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()  # Zera os gradientes do otimizador
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    return model
   
# Funçao para fazer a previsao entre dois ou mais sonhos
def predict_similarity(model, vectorizer, dream1, dream2):
    # Pre-processa os dois sonhos 
    dream1_processed = preprocess_text(dream1)
    deram2_processed = preprocess_text(dream2)

    # Transforma os sonhso em vetores para a rede entender (TF-IDF)
    dreams = [dream1_processed, deram2_processed]
    X = vectorizer.tranform(dreams).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32) # Converte os vetores para pyTorch

    # Autograd está ativa aqui, então os gradientes serão calculados(Ela deixa o script mais pesado e nao melhora a propabilidade, pore, eu usava antes)
   # output = model(X_tensor[0] - X_tensor[1])
    #return output.item()


    with torch.no_grad(): # Desativamos a autograd para inferencia
        output = model(X_tensor[0] - X_tensor[1]) # Passa a diferença entre os valores pela rede

    return output.item() # Retorna a saida como um valor escalar

# Funçao para ler os sonhos atraves de um arquivo txt
def read_dreams_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dreams = file.readlines()  # Lê todas as linhas do arquivo como uma lista
    dreams = [dream.strip() for dream in dreams]  # Remove espaços em branco no início e fim de cada linha
    return dreams  # Retorna a lista de sonhos

# Arquivo que tera os sonhos escritos 
dreams = read_dreams_from_txt('sonhos.txt') # Le os sonhos desse arquivo

relations = [0, 1, 0] # Relaçoes entre sonhos (1: relacionados, 0: nao relacionados)

# Instancia o vetor TF-IDF e pre-processa os sonhos
vectorizer = TfidfVectorizer() 
dreams_processed = [preprocess_text(dream) for dream in dreams]
X = vectorizer.fit_transform(dreams_processed).toarray() # Concerte os sonhos em vetores TF-IDF

model = train_model(X, np.array(relations)) # Treina o modelo com os dados

dream1 = 'Eu tava dando mó beijo nela' # Novo sonho 1
dream2 = 'Eu tava comendo um cachorro quente' # Novo sonho 2

# Predi a similaridade entre os dois novos sonhos 
similarity = predict_similarity(model, vectorizer, dream1, dream2)
print(f'Similiaridade entre os dois sonhos: {similarity:.4f}') # Imprume a similaridade 