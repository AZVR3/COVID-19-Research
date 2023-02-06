import pandas as pd
from prophet import Prophet

# Read the data from 'south-korea-data.csv' into a dataframe
df = pd.read_csv('south-korea-data.csv')

# Rename the columns to be consistent with Prophet's expected input format
df = df.rename(columns={'Date_reported': 'ds', 'Cumulative_cases': 'y'})

# Filter the data to only include dates from 9/14/2022 to 10/29/2022
df = df[(df['ds'] >= '2022-09-14') & (df['ds'] <= '2022-10-29')]

# Initialize the Prophet model
model = Prophet()

# Fit the model on the data
model.fit(df)

# Make predictions for the dates from 10/30/2022 to 1/30/2023
future = model.make_future_dataframe(periods=91, freq='D', start='2022-10-30', end='2023-01-30')
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
