import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer


def evaluate_model(model_path="../models/bert-sentiment"):
    dataset = load_dataset("imdb")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=256)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer = Trainer(model=model)

    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    print(classification_report(labels, preds))

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title("Confusion Matrix")
    plt.show()

    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:,1]
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
