from fastapi import FastAPI
import torch

from data import logs
from preprocess import encode_sequence, encoder
from model import LogModel

app = FastAPI()


X = [encode_sequence(seq) for seq in logs]
max_len = max(len(seq) for seq in X)


vocab_size = len(encoder.classes_)
model = LogModel(vocab_size)

model.load_state_dict(torch.load("model.pth"))
model.eval()



def predict(seq):
    if not seq:
        return {
            "error": "No logs provided"
        }

    encoded = encode_sequence(seq)

   
    if len(encoded) < max_len:
        padded = encoded + [0] * (max_len - len(encoded))
    else:
        padded = encoded[:max_len]  

    tensor = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        prob = model(tensor).item()

    return {
        "prediction": "ANOMALY" if prob > 0.5 else "NORMAL",
        "confidence": prob
    }



@app.post("/predict")
def predict_api(data: dict):
    logs_input = data.get("logs", [])
    return predict(logs_input)