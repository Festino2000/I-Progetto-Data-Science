import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. ETL e Preparazione dei dati
print("Caricamento dataset in corso...")
df = pd.read_csv('train.csv')
df['date'] = pd.to_datetime(df['date'])
vendite_giornaliere = df.groupby('date')['sales'].sum().reset_index()

df_serie = vendite_giornaliere.set_index('date')
df_serie = df_serie.asfreq('D').fillna(0)

# 2. Test di stazionarietà ADF
print("Esecuzione Test ADF...")
result = adfuller(df_serie['sales'].dropna())
print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4e}")

# 3. Train / Test Split (80% / 20%)
serie_prezzi = df_serie['sales']
train_size = int(len(serie_prezzi) * 0.8)
train_data = serie_prezzi[:train_size]
test_data = serie_prezzi[train_size:]

# 4. Addestramento Modello SARIMAX Stagionale (s=7)
print("Addestramento SARIMAX in corso...")
modello_sarimax = SARIMAX(train_data,
                          order=(1, 1, 1),
                          seasonal_order=(1, 1, 1, 7),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
risultati = modello_sarimax.fit(disp=False)

# 5. Previsioni e Metriche
previsioni = risultati.predict(start=test_data.index[0], end=test_data.index[-1])
mae = mean_absolute_error(test_data, previsioni)
rmse = np.sqrt(mean_squared_error(test_data, previsioni))

print(f"Risultati finali -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")