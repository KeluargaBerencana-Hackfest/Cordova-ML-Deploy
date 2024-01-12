import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

print(BASE_DIR)
with open(f"{BASE_DIR}/trained_model_rfr.pkl", "rb") as f:
    model = pickle.load(f)
    
def predict(data):
    result = model.predict(data)
    return result[0] * 100