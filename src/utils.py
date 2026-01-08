from transformers import BertTokenizer

def load_tokenizer(model_path="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_path)


def preprocess_text(tokenizer, text, max_length=256):
    return tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
