# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 4-11-25
### NAME: JESUBALAN A
### REG_NO: 212223240060

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

data = pd.read_csv('TSLA.csv', parse_dates=['Date'], index_col='Date')
tsla_data = data[['Close']]

plt.figure(figsize=(12, 6))
plt.plot(tsla_data['Close'], label='Original Close Price Data')
plt.title('Tesla Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = tsla_data['Close'].rolling(window=5).mean()
rolling_mean_10 = tsla_data['Close'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(tsla_data['Close'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Tesla Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

data_monthly = tsla_data.resample('MS').mean()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

scaled_data = scaled_data + 1e-6

x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(12, 6))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["Train Data", "Predictions", "Test Data"])
ax.set_title('Train vs Test Forecast (Tesla Stock)')
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("RMSE:", rmse)

model_final = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model_final.forecast(steps=12)

ax = scaled_data.plot(figsize=(12, 6))
predictions.plot(ax=ax)
ax.legend(["Scaled Data", "Forecast (Next Year)"])
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Close Price')
ax.set_title('Tesla Stock Price Forecast')
plt.show()

```
### OUTPUT  :

### Plot Transform Dataset :
<img width="1204" height="608" alt="Screenshot 2025-11-19 191747" src="https://github.com/user-attachments/assets/e80702ce-dc38-4326-8562-0df5541555e0" />

<img width="1220" height="619" alt="Screenshot 2025-11-19 191811" src="https://github.com/user-attachments/assets/ad1869a6-f7a2-4832-899c-82819c70987b" />

<img width="1207" height="662" alt="Screenshot 2025-11-19 191829" src="https://github.com/user-attachments/assets/fd6863bc-dcbc-4b22-8dee-ba3c693b8ff7" />

<img width="1163" height="623" alt="Screenshot 2025-11-19 191845" src="https://github.com/user-attachments/assets/e9422a1e-ca02-48cd-bd0c-0d16230a10a1" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
