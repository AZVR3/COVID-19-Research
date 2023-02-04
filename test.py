import pandas as pd

df = pd.read_csv("WHO-COVID-19-global-data.csv")
print("Shape of 'Date_reported' column:", df['Date_reported'].shape)
print("Shape of 'Cumulative_cases' column:", df['Cumulative_cases'].shape)
