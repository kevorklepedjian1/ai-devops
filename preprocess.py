from sklearn.preprocessing import LabelEncoder
from data import logs


all_logs = [log.lower() for seq in logs for log in seq]

encoder = LabelEncoder()
encoder.fit(all_logs)

def encode_sequence(seq):
    seq = [log.lower() for log in seq]
    return encoder.transform(seq).tolist()

def decode_sequence(seq):
    return encoder.inverse_transform(seq)