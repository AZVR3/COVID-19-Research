import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('south-korea-data.csv', parse_dates=['Date_reported'], index_col='Date_reported')
data_train = data.loc['2022-09-14':'2022-10-29'].reset_index()
data_train = data_train.rename(columns={'Date_reported': 'ds', 'Cumulative_cases': 'y'})

# Train Prophet model
model = Prophet()
model.fit(data_train)

# Make predictions
start_date = '2022-10-30'
end_date = '2023-01-30'
data_test = data.loc[start_date:end_date].reset_index()
data_test = data_test.rename(columns={'Date_reported': 'ds', 'Cumulative_cases': 'y'})
future = model.make_future_dataframe(periods=len(data_test))
predictions = model.predict(future)
predictions = predictions.loc[(predictions['ds'] >= start_date) & (predictions['ds'] <= end_date), 'yhat']

# Calculate errors
mse = mean_squared_error(data_test['y'], predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(data_test['y'], predictions)
mape = np.mean(np.abs(predictions - data_test['y']) / np.abs(data_test['y'])) * 100

# Print errors
print('RMSE:', rmse)
print('MAE:', mae)
print('MAPE:', mape)

# Plot actual vs predicted values
plt.plot(data_test['ds'], data_test['y'], label='Actual')
plt.plot(data_test['ds'], predictions, label='Predicted')
plt.legend()
plt.show()
