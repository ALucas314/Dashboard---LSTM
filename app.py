import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import sqlalchemy

# Configurações iniciais
st.set_page_config(page_title="Clima e Previsões", layout="wide", page_icon="🌤️")

# Função para carregar dados do banco de dados
@st.cache_data
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
        st.error(f"Erro ao carregar os dados: {err}")
        return None

# Função para carregar o modelo de um arquivo .pkl
@st.cache_resource
def carregar_modelo(arquivo_pkl):
    if os.path.exists(arquivo_pkl):
        with open(arquivo_pkl, 'rb') as file:
            return pickle.load(file)
    return None

# Sidebar - Configurações
st.sidebar.title("Configurações")
st.sidebar.markdown("### Personalize sua visualização de dados")

# Carregar dados do banco de dados
data = carregar_dados()

if data is not None and not data.empty:
    st.title("🌤️ Dados Meteorológicos - Castanhal")
    st.markdown("Explore os dados meteorológicos com previsões baseadas em redes neurais LSTM.")

    # Sidebar - Escolher o número de linhas para exibir
    linhas = st.sidebar.slider("Escolha o número de linhas para exibir:", min_value=5, max_value=len(data), value=10)

    # Exibir tabela de dados
    st.subheader("Tabela de Dados")
    st.dataframe(data.head(linhas), use_container_width=True)

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

    # Verificar se o modelo já foi salvo
    model_data = carregar_modelo('pesoLSTM.pkl')
    if model_data:
        st.sidebar.write("Carregando modelo salvo...")
        model = model_data['model']
        scaler = model_data['scaler']
    else:
        st.sidebar.write("Modelo não encontrado. Treine o modelo primeiro.")
        st.stop()

    # Previsões
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform([y_test])

    # Média da temperatura
    media_temperatura = weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].mean()
    st.sidebar.write(f'Média da temperatura: {media_temperatura:.2f}°C')

    # Cálculo do MSE (Erro Quadrático Médio)
    mse_lstm = mean_squared_error(y_test_rescaled[0], y_pred_rescaled)
    st.sidebar.write(f'Mean Squared Error (MSE) - LSTM: {mse_lstm:.4f}')

    # Previsão para amanhã (próxima temperatura)
    last_data = data_scaled[-time_step:].reshape(1, time_step, 1)
    prediction = model.predict(last_data)
    prediction_rescaled = scaler.inverse_transform(prediction)
    st.sidebar.write(f'Previsão de Temperatura para Amanhã: {prediction_rescaled[0][0]:.2f}°C')

    # Layout dos gráficos
    st.subheader("Visualizações Gráficas")

    # Dividindo a tela em 2 colunas para os gráficos lado a lado
    col1, col2 = st.columns(2)

    # Gráfico de previsão - Comparação entre Temperatura Real e Previsões LSTM
    with col1:
        st.markdown("### Comparação entre Temperatura Real e Previsões LSTM")
        fig = plt.figure(figsize=(8, 4))  # Tamanho menor
        plt.plot(y_test_rescaled[0], label='Temperatura Real', color='blue', linewidth=2)
        plt.plot(y_pred_rescaled, label='Temperatura Prevista (LSTM)', color='red', linestyle='--', linewidth=2)
        plt.title('Temperatura Real vs Previsões LSTM', fontsize=14)
        plt.xlabel('Índice de Teste', fontsize=10)
        plt.ylabel('Temperatura (°C)', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Gráfico de Temperatura ao Longo do Tempo (Agora com número do dia)
    with col2:
        st.markdown("### Temperatura ao Longo do Tempo")
        fig_temp = plt.figure(figsize=(8, 4))  # Tamanho adequado para ajustar
        # Usando apenas o número do dia
        days = weather_data['Data'].dt.day
        plt.plot(days, weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'], label='Temperatura', color='green', linewidth=2)
        plt.title('Temperatura ao Longo do Tempo', fontsize=14)
        plt.xlabel('Dia', fontsize=12)
        plt.ylabel('Temperatura (°C)', fontsize=12)
        # Agora garantindo que os números dos dias sejam horizontais
        plt.xticks(rotation=0)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_temp)

    # Gráfico abaixo: Temperatura ao Longo do Tempo com Média
    st.markdown("### Temperatura ao Longo do Tempo com Média")
    fig_temp_media = plt.figure(figsize=(8, 4))  # Tamanho adequado para ajustar
    media_temperatura = weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].mean()
    plt.plot(days, weather_data['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'], label='Temperatura', color='green', linewidth=2)
    plt.axhline(y=media_temperatura, color='orange', linestyle='--', label=f'Média: {media_temperatura:.2f}°C')
    plt.title('Temperatura ao Longo do Tempo com Média', fontsize=14)
    plt.xlabel('Dia', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    # Agora garantindo que os números dos dias sejam horizontais
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_temp_media)

else:
    st.warning("Nenhum dado disponível para exibição.")
