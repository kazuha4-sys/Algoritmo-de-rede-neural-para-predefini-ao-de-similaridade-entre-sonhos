import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Função para criar e treinar o modelo
def train_and_save_model():
    # Dados de exemplo
    X = np.array([
        [20, 30, 50, 10, 15, 25],
        [15, 25, 55, 5, 20, 30],
        [30, 20, 40, 20, 10, 25]
    ])
    
    y = np.array([20.0, 15.0, 25.0])

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Criação e treinamento do modelo
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Salvando o modelo e o scaler
    joblib.dump(model, 'dream_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Modelo e scaler salvos com sucesso!")

# Função para prever a porcentagem com base em novos dados
def predict_percentage(past_dreams, current_dreams):
    # Carregar o modelo e o scaler
    model = joblib.load('dream_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Preparar dados de entrada
    input_data = np.array([
        past_dreams['fuga'], past_dreams['felicidade'], past_dreams['medo'],
        current_dreams['fuga'], current_dreams['felicidade'], current_dreams['medo']
    ]).reshape(1, -1)
    
    # Normalizar os dados
    input_data_scaled = scaler.transform(input_data)
    
    # Fazer a previsão
    prediction = model.predict(input_data_scaled)
    return prediction[0]

if __name__ == '__main__':
    # Treinar e salvar o modelo
    train_and_save_model()

    # Exemplo de uso da função de previsão
    past_dreams = {'fuga': 20, 'felicidade': 30, 'medo': 50}
    current_dreams = {'fuga': 10, 'felicidade': 15, 'medo': 25}
    percentage = predict_percentage(past_dreams, current_dreams)
    print(f"Predição da porcentagem de sonhos de fuga: {percentage:.2f}%")
