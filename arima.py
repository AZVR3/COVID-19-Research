import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# load data and set the index to be the date
data = pd.read_csv('WHO-COVID-19-global-data.csv', index_col='Date_reported', parse_dates=True)

# differentiate the data until it reaches a stationary state
differenced_data = data.diff().dropna()

# plot the stationary data
plt.plot(differenced_data)
plt.xlabel('Date')
plt.ylabel('Differenced Total cases')
plt.show()

# fit an ARIMA model to the time series
model = ARIMA(differenced_data, order=(3, 1, 2))
model_fit = model.fit()

# make predictions for the next 100 days
predictions = model_fit.forecast(steps=100)[0]

# convert predictions to a DataFrame
predictions = pd.DataFrame(predictions, index=pd.date_range(start=differenced_data.index[-1], end=differenced_data.index[-1]+pd.Timedelta(days=100), freq='D'), columns=['Total cases'])

# cumulative sum to obtain the original values
predictions = predictions.cumsum().shift(1).fillna(0)
predictions[data.columns[0]] = predictions[predictions.columns[0]] + data.iloc[-1][0]

# plot the forecasted values along with the original data
plt.plot(data)
plt.plot(predictions, color='red')
plt.xlabel('Date')
plt.ylabel('Total cases')
plt.show()
