import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import sqlalchemy

# Função para carregar dados do banco de dados
def carregar_dados():
    try:
        engine = sqlalchemy.create_engine("mysql+mysqlconnector://lucas:456321@localhost/clima")
        query = "SELECT * FROM castanhal"
        data = pd.read_sql(query, engine)
        # Verificar e corrigir a coluna "Data"
        if "Data" in data.columns:
            data["Data"] = pd.to_datetime(data["Data"], format="%Y/%m/%d", errors="coerce")
            data = data.dropna(subset=["Data"])
        return data
    except Exception as err:
        print(f"Erro ao carregar os dados: {err}")
        return None

# Função para salvar o modelo em um arquivo .pkl
def salvar_modelo(model, scaler, arquivo_pkl):
    with open(arquivo_pkl, 'wb') as file:
        pickle.dump({'model': model, 'scaler': scaler}, file)

# Função para treinar o modelo
def treinar_modelo():
    # Carregar os dados
    data = carregar_dados()

    if data is None or data.empty:
        print("Erro: Nenhum dado disponível para treinamento.")
        return
    
    # Pré-processamento dos dados
    weather_data = data[['Data', 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)']]
    weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] = weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].str.replace(',', '.')
    weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] = pd.to_numeric(weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'], errors='coerce')
    weather_data = weather_data.dropna(subset=['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'])
    
    # Ordenar os dados por data
    weather_data = weather_data.sort_values(by='Data')

    data = weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 100
    X, y = create_dataset(data_scaled, time_step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Criar o modelo LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(units=120, return_sequences=True), input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=70, batch_size=32, verbose=1)

    # Salvar o modelo treinado
    salvar_modelo(model, scaler, 'pesoLSTM.pkl')
    print("Modelo treinado e salvo com sucesso!")

if __name__ == "__main__":
    treinar_modelo()
