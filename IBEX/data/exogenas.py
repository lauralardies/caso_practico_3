import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

start_date = '2012-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Descargar datos históricos de diferentes activos
ibex35 = yf.download('^IBEX', start=start_date, end=end_date)
sp500 = yf.download('^GSPC', start=start_date, end=end_date)
petroleo = yf.download('CL=F', start=start_date, end=end_date)
eur_usd = yf.download('EURUSD=X', start=start_date, end=end_date)

# Renombrar las columnas para hacerlas más entendibles
ibex35 = ibex35.rename(columns={'Adj Close': 'IBEX35_AdjClose', 'Open': 'IBEX35_Open', 'High': 'IBEX35_High', 'Low': 'IBEX35_Low', 'Close': 'IBEX35_Close', 'Volume': 'IBEX35_Volume'})
sp500 = sp500.rename(columns={'Adj Close': 'S&P500_AdjClose', 'Open': 'S&P500_Open', 'High': 'S&P500_High', 'Low': 'S&P500_Low', 'Close': 'S&P500_Close', 'Volume': 'S&P500_Volume'})
petroleo = petroleo.rename(columns={'Adj Close': 'Petróleo WTI_AdjClose', 'Open': 'Petroleo_Open', 'High': 'Petroleo_High', 'Low': 'Petroleo_Low', 'Close': 'Petroleo_Close', 'Volume': 'Petroleo_Volume'})
eur_usd = eur_usd.rename(columns={'Adj Close': 'EUR/USD_AdjClose', 'Open': 'EUR/USD_Open', 'High': 'EUR/USD_High', 'Low': 'EUR/USD_Low', 'Close': 'EUR/USD_Close', 'Volume': 'EUR/USD_Volume'})

# Unir los DataFrames en uno solo, alineando las fechas (usamos 'inner' para solo quedarnos con las fechas comunes)
df = ibex35.join([sp500, petroleo, eur_usd], how='inner')

# Guardar el DataFrame resultante en un archivo CSV
df.to_csv('./IBEX/data/datos_completos_exogenos_ibex35.csv')

print("Archivo CSV guardado exitosamente.")