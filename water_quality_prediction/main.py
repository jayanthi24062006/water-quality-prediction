import pandas as pd

# Load dataset
df = pd.read_csv('water_potability.csv')

# Display first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# Display missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())