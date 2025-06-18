import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# --- Page Config ---
st.set_page_config(page_title="Stock Forecasting", layout="wide")
st.title("📈 Time Series Stock Forecasting App")

# --- Input ---
ticker = st.text_input("Enter stock ticker", "AAPL")
years = st.slider("Years to forecast", 1, 5)
period = years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="10y")
    data.reset_index(inplace=True)
    return data[['Date', 'Close']]

df = load_data(ticker)

# --- Plot Historical ---
st.subheader("Historical Closing Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Closing Price"))
st.plotly_chart(fig, use_container_width=True)

# --- Prophet Forecast ---
st.header("🔮 Prophet Forecast")
df_p = df.rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_p)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast"))
fig1.update_layout(title="Prophet Forecast")
st.plotly_chart(fig1, use_container_width=True)

# --- ARIMA Forecast ---
st.header("🔢 ARIMA Forecast")
model_arima = ARIMA(df['Close'], order=(5, 1, 0))
fit_arima = model_arima.fit()
forecast_arima = fit_arima.forecast(steps=period)
forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=period+1, freq='D')[1:]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_dates, y=forecast_arima, name="ARIMA"))
fig2.update_layout(title="ARIMA Forecast")
st.plotly_chart(fig2, use_container_width=True)

# --- LSTM Forecast ---
st.header("🧠 LSTM Forecast")
data = df[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X_train, y_train = [], []
look_back = 60
for i in range(look_back, len(scaled_data)):
    X_train.append(scaled_data[i-look_back:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

last_sequence = scaled_data[-look_back:]
predictions = []

for _ in range(period):
    pred = model_lstm.predict(last_sequence.reshape(1, look_back, 1))[0][0]
    predictions.append(pred)
    last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)

predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=period+1, freq='D')[1:]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=future_dates, y=predicted_prices.flatten(), name="LSTM"))
fig3.update_layout(title="LSTM Forecast")
st.plotly_chart(fig3, use_container_width=True)