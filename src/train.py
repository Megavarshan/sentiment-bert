from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def main():
    dataset = load_dataset("imdb")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=256)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="../models/bert-sentiment",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="../logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    trainer.save_model("../models/bert-sentiment")
    tokenizer.save_pretrained("../models/bert-sentiment")


if __name__ == "__main__":
    main()
