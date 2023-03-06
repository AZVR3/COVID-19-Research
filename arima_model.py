import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('south-korea-data.csv', parse_dates=['Date_reported'], index_col='Date_reported')
data_train = data.loc['2022-09-14':'2022-10-29']['Cumulative_cases']
# Test 3 (3 months):

# Test 2 (9 months):
# data_train = '2022-01-14':'2022-10-29'
# RMSE: 906592.4645111117
# MAE: 752372.3286691037
# MAPE: 2.5992858099382943

# Test 1 (1 months):
# data_train = '2022-09-14':'2022-10-29'
# prediction = '2022-10-30': '2023-01-30'
# RMSE: 890718.4113883465
# MAE: 738278.4935086957
# MAPE: 2.550284189897427

# Train ARIMA model
model = ARIMA(data_train, order=(2,2,4))
model_fit = model.fit()

# Make predictions
start_date = '2022-10-30'
end_date = '2023-01-30'
data_test = data.loc[start_date:end_date]['Cumulative_cases']
predictions = model_fit.predict(start=start_date, end=end_date, typ='levels')
# Test 1 (3 months):

# Calculate errors
mse = mean_squared_error(data_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(data_test, predictions)
mape = np.mean(np.abs(predictions - data_test) / np.abs(data_test)) * 100

# Print errors
print('RMSE:', rmse)
print('MAE:', mae)
print('MAPE:', mape)

# Print model summary
print(model_fit.summary())

# Plot actual vs predicted values
plt.plot(data_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
