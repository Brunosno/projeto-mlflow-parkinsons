import urllib.request
import os

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
    os.makedirs("data", exist_ok=True)
    filepath = "data/parkinsons_updrs.data"
    if not os.path.exists(filepath):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, filepath)
        print("Dataset downloaded.")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_data()
