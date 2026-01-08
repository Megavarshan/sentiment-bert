import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import preprocess_text, load_tokenizer

def predict_sentiment(text, model_path="../models/bert-sentiment"):
    tokenizer = load_tokenizer(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    tokens = preprocess_text(tokenizer, text)
    output = model(**tokens)
    
    label = torch.argmax(output.logits).item()
    return "Positive" if label == 1 else "Negative"


if __name__ == "__main__":
    sample = "I absolutely loved this movie!"
    print("Input:", sample)
    print("Prediction:", predict_sentiment(sample))
