import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('south-korea-data.csv')
# df.info()
df = np.log(df)

df.plot()
plt.show()