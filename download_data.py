import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ticker = "AAPL"
data = yf.download(ticker, start="2018-01-01", end="2023-01-01")
print(data.head())
data = data.dropna()
data.index = pd.to_datetime(data.index)



#Line graph of closed prices
data['Close'].plot(figsize=(10, 6))
plt.title('Apple Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()



# Histogram of trading volume
sns.histplot(data['Volume'])
plt.title('Distribution of Trading Volume')
plt.xlabel('Volume')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
sns.heatmap(data.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()


#Destroyed price value closed
data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag2'] = data['Close'].shift(2)
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA10'] = data['Close'].rolling(window=10).mean()
data = data.dropna()



# Linear regression
X = data[['Close_Lag1', 'Close_Lag2', 'MA5', 'MA10']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# Future price forecast
last_close = data['Close'].iloc[-1]
last_close_lag1 = data['Close'].iloc[-2]
last_close_lag2 = data['Close'].iloc[-3]
last_ma5 = data['MA5'].iloc[-1]
last_ma10 = data['MA10'].iloc[-1]
new_data = [[last_close, last_close_lag1, last_close_lag2, last_ma5]]
future_price = model.predict(new_data)
print(f'Predicted Future Price: {future_price[0]}')
