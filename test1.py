import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data from 'south-korea-data.csv'
df = pd.read_csv('south-korea-data.csv')

# Convert date column to datetime
df['Date_reported'] = pd.to_datetime(df['Date_reported'])
df.set_index('Date_reported', inplace=True)

# Filter data range from 9/14/2022 ~ 10/29/2022
start_date = datetime(2022, 9, 14)
end_date = datetime(2022, 10, 29)
df = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date)]

# Fit an ARIMA model
model = sm.tsa.ARIMA(df['Cumulative_cases'], order=(1,1,1))
arima_fit = model.fit()

# Forecast 10/30/2022 ~ 1/30/2023
forecast = arima_fit.forecast(steps=91)
forecast_df = pd.DataFrame({'forecast': forecast[0].data}, index=forecast[0].index)


# Plot the forecasted data
plt.plot(forecast_df['forecast'], label='Forecast')
plt.legend()
plt.show()
