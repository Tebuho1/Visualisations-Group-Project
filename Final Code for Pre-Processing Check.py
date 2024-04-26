import pandas as pd
import numpy as np

# Load a DataFrame from the CSV dataset
df = pd.read_csv('customer_shopping_data.csv')

# Check for missing values in the dataset
df_missing = df.isnull().sum()
if df_missing.sum() > 0:
    print('Missing values in each column:', df_missing)
else:
    print('No missing values found.')

# Check for any duplicate rows
duplicates = df.duplicated().sum()
if duplicates > 0:
    print('Number of duplicate rows:', duplicates)
else:
    print('No duplicate rows found.')

# Check for null values in the dataset
df_null = df.isnull().sum()
if df_null.sum() > 0:
    print('Null values in each column:', df_null)
else:
    print('No null values found.')
    
