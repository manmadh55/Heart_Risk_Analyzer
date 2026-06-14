import urllib.request
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
filename = "processed.cleveland.data"

try:
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")
except Exception as e:
    print("Error downloading:", e)

# Test reading
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

df = pd.read_csv(filename, header=None, names=column_names, na_values='?')
print("Shape:", df.shape)
print("Nulls:\n", df.isnull().sum())
print("Target distribution:\n", df['num'].value_counts())
