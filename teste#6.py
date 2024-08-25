# Esse ele usa dados antigos para treinamento, ou sjea, ele vai usar dados de treinos antigos para ajudar treinar
import torch  # Importa a biblioteca PyTorch para operações de aprendizado profundo
import torch.nn as nn  # Importa o módulo de redes neurais do PyTorch
import torch.optim as optim  # Importa otimizadores do PyTorch
from sklearn.feature_extraction.text import TfidfVectorizer  # Importa o vetorizador TF-IDF para processamento de texto
from sklearn.preprocessing import StandardScaler  # Importa o escalador padrão para normalização de dados
from nltk.corpus import stopwords  # Importa a lista de palavras de parada do NLTK
from nltk.tokenize import word_tokenize  # Importa o tokenizador de palavras do NLTK
import nltk  # Importa a biblioteca NLTK
import numpy as np  # Importa a biblioteca NumPy para operações matemáticas

# Baixando recursos necessários do NLTK
nltk.download('punkt')  # Baixa o conjunto de dados para tokenização de palavras
nltk.download('stopwords')  # Baixa o conjunto de dados para palavras de parada

# Definição da Rede Neural com Regularização e Normalização
class DreamNetwork(nn.Module):  # Define uma nova classe chamada DreamNetwork que herda de nn.Module
    def __init__(self, input_dim):  # Método de inicialização que define a arquitetura da rede
        super(DreamNetwork, self).__init__()  # Inicializa a classe base nn.Module
        self.fc1 = nn.Linear(input_dim, 128)  # Camada totalmente conectada com 128 neurônios e input_dim entradas
        self.dropout1 = nn.Dropout(0.5)  # Dropout para regularização
        self.fc2 = nn.Linear(128, 64)  # Outra camada totalmente conectada com 64 neurônios
        self.dropout2 = nn.Dropout(0.5)  # Outra camada de dropout para regularização
        self.fc3 = nn.Linear(64, 1)  # Camada de saída com 1 neurônio (para uma tarefa binária)
    
    def forward(self, x):  # Método de propagação direta que define o fluxo de dados na rede
        x = torch.relu(self.fc1(x))  # Aplica a função de ativação ReLU na primeira camada
        x = self.dropout1(x)  # Aplica dropout após a primeira camada
        x = torch.relu(self.fc2(x))  # Aplica a função de ativação ReLU na segunda camada
        x = self.dropout2(x)  # Aplica dropout após a segunda camada
        x = torch.sigmoid(self.fc3(x))  # Aplica a função de ativação sigmoid na camada de saída para obter uma probabilidade entre 0 e 1
        return x

# Pré-processamento de texto
def preprocess_text(dream):  # Função para pré-processar o texto dos sonhos
    stop_words = set(stopwords.words('portuguese'))  # Obtém a lista de palavras de parada em português
    tokens = word_tokenize(dream.lower())  # Tokeniza o texto e converte para minúsculas
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]  # Filtra palavras alfanuméricas e remove palavras de parada
    return ' '.join(filtered_words)  # Retorna o texto filtrado como uma string

# Função para normalizar os dados
def normalize_data(X):  # Função para normalizar os dados usando StandardScaler
    scaler = StandardScaler()  # Cria uma instância do StandardScaler
    return scaler.fit_transform(X)  # Ajusta o escalador e transforma os dados

# Função para treinar o modelo com Early Stopping
def train_model(X, y, model=None):  # Função para treinar a rede neural com dados e rótulos; permite continuar o treinamento de um modelo existente
    input_dim = X.shape[1]  # Número de características (dimensão de entrada)
    if model is None:  # Se não houver um modelo pré-existente
        model = DreamNetwork(input_dim)  # Cria uma nova instância do modelo de rede neural
    criterion = nn.BCELoss()  # Define a função de perda para problemas de classificação binária
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define o otimizador Adam com taxa de aprendizado de 0.001
    
    # Converte os dados e rótulos para tensores do PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Verificação do tamanho dos tensores
    if X_tensor.size(0) != y_tensor.size(0):  # Verifica se o número de amostras é o mesmo
        raise ValueError(f"Tamanhos incompatíveis: X_tensor tem {X_tensor.size(0)} amostras e y_tensor tem {y_tensor.size(0)} amostras.")
    
    best_loss = float('inf')  # Inicializa a melhor perda como infinito
    patience = 10  # Número de épocas para aguardar sem melhoria antes de parar
    patience_counter = 0  # Contador para Early Stopping
    
    for epoch in range(100):  # Loop de treinamento por 100 épocas
        model.train()  # Coloca o modelo em modo de treinamento
        optimizer.zero_grad()  # Zera os gradientes do otimizador
        outputs = model(X_tensor)  # Faz a previsão
        
        # Verificação do tamanho da saída
        if outputs.size() != y_tensor.size():  # Verifica se o tamanho da saída corresponde ao tamanho dos rótulos
            raise ValueError(f"Tamanhos incompatíveis: outputs tem {outputs.size()} e y_tensor tem {y_tensor.size()}.")
        
        loss = criterion(outputs, y_tensor)  # Calcula a perda
        loss.backward()  # Calcula os gradientes
        optimizer.step()  # Atualiza os parâmetros do modelo
        
        # Early Stopping
        if loss.item() < best_loss:  # Se a perda atual for menor que a melhor perda
            best_loss = loss.item()  # Atualiza a melhor perda
            patience_counter = 0  # Reseta o contador de paciência
        else:
            patience_counter += 1  # Incrementa o contador de paciência
        
        if patience_counter >= patience:  # Se o contador de paciência exceder o valor definido
            print(f'Early stopping at epoch {epoch+1}')  # Imprime mensagem de parada antecipada
            break
        
        if (epoch + 1) % 10 == 0:  # A cada 10 épocas
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')  # Imprime a perda

    return model  # Retorna o modelo treinado

# Função para prever a similaridade entre dois sonhos
def predict_similarity(model, vectorizer, dream1, dream2):  # Função para prever a similaridade entre dois sonhos
    dream1_processed = preprocess_text(dream1)  # Processa o texto do primeiro sonho
    dream2_processed = preprocess_text(dream2)  # Processa o texto do segundo sonho
    
    # Vetoriza os sonhos processados
    dreams = [dream1_processed, dream2_processed]
    X = vectorizer.transform(dreams).toarray()  # Converte os textos em matrizes TF-IDF
    X = normalize_data(X)  # Normaliza os dados
    X_tensor = torch.tensor(X, dtype=torch.float32)  # Converte para tensor do PyTorch
    
    # Autograd está ativa aqui, então os gradientes serão calculados
    output = model(X_tensor[0] - X_tensor[1])  # Calcula a saída do modelo para a diferença dos vetores de sonhos
    
    return output.item()  # Retorna o valor da similaridade como um número

# Função para carregar sonhos de um arquivo .txt
def load_dreams_from_file(filename):  # Função para carregar sonhos de um arquivo de texto
    with open(filename, 'r', encoding='utf-8') as file:  # Abre o arquivo com codificação UTF-8
        dreams = [line.strip() for line in file.readlines()]  # Lê todas as linhas e remove espaços extras
    return dreams  # Retorna a lista de sonhos

# Exemplo de uso
dreams = load_dreams_from_file('sonhos.txt')  # Carrega os sonhos do arquivo 'sonhos.txt'

# Atualize as relações para corresponder ao número de sonhos
# Cada par de sonhos deve ter uma relação correspondente
# Para simplificar, vou usar um exemplo onde todos os pares são relacionados (1)
relations = [1] * len(dreams)  # Exemplo: Todos os pares são relacionados

# Verifica se o número de relações corresponde ao número de sonhos
if len(dreams) != len(relations):  # Verifica se o número de sonhos e relações é o mesmo
    raise ValueError(f"O número de sonhos ({len(dreams)}) deve corresponder ao número de relações ({len(relations)}).")

vectorizer = TfidfVectorizer()  # Cria uma instância do vetorizador TF-IDF
dreams_processed = [preprocess_text(dream) for dream in dreams]  # Processa todos os sonhos
X = vectorizer.fit_transform(dreams_processed).toarray()  # Transforma os sonhos em matrizes TF-IDF e converte para array
X = normalize_data(X)  # Normaliza os dados

# Verifica se existe um modelo treinado salvo; se não, treina um novo modelo
try:
    model = torch.load('dados_treinamento/dream_network_model.pth')  # Tenta carregar um modelo previamente treinado
    print("Modelo carregado com sucesso!")
except FileNotFoundError:  # Se o modelo não for encontrado
    model = train_model(X, np.array(relations))  # Treina um novo modelo com os dados e relações
    torch.save(model, 'dados_treinamento/dream_network_model.pth')  # Salva o modelo treinado em um arquivo

dream1 = "Eu estava voando no céu"  # Exemplo de sonho 1
dream2 = "Eu estava caindo de um prédio alto"  # Exemplo de sonho 2

similarity = predict_similarity(model, vectorizer, dream1, dream2)  # Prevejo a similaridade entre os dois sonhos
print(f"Similaridade entre os sonhos: {similarity:.4f}")  # Imprime a similaridade
