import numpy as np
import pandas as pd

df = pd.read_csv('data/raw/assurance-maladie.csv')
df.info()
df.describe()
df['sex'].value_counts()
df['smoker'].value_counts()
df['region'].value_counts()
