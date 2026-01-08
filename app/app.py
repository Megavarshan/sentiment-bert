import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("../models/bert-sentiment")
model = BertForSequenceClassification.from_pretrained("../models/bert-sentiment")

def predict(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = model(**tokens)
    label = torch.argmax(output.logits).item()
    return "Positive 😀" if label == 1 else "Negative 😠"

app = gr.Interface(fn=predict, inputs="text", outputs="text", title="BERT Sentiment Classifier")
app.launch()
