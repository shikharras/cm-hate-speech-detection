import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

NUM_RUNS = 5
CHECKPOINT = "xlm-roberta-base"
# l3cube-pune/hing-roberta, l3cube-pune/hing-roberta-mixed, xlm-roberta-base

model_name = CHECKPOINT.split("/")[-1]
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_steps=50,
        evaluation_strategy="epoch", #steps
        label_names=["labels"]
    )

class HSDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_datasets(tokenizer, train_df, val_df, test_df):
    train_encodings = tokenizer(train_df['tweet_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    val_encodings = tokenizer(val_df['tweet_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    test_encodings = tokenizer(test_df['tweet_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

    train_dataset = HSDataset(train_encodings, train_df['offense'].tolist())
    val_dataset = HSDataset(val_encodings, val_df['offense'].tolist())
    test_dataset = HSDataset(test_encodings, test_df['offense'].tolist())
    return train_dataset, val_dataset, test_dataset

def get_class_weights(train_df):
    class_instance_counts = torch.tensor(list(dict(Counter(train_df['offense'].tolist())).values()))
    class_instance_probs = class_instance_counts/class_instance_counts.sum().item()
    class_weights = 1 / class_instance_probs
    class_weights.to(device)
    return class_weights

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def get_trainer(model, train_dataset, val_dataset, class_weights):
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='binary')
        return {'f1': f1}

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights=class_weights.to(device),
        compute_metrics = compute_metrics
    )
    return trainer

def get_test_results(trainer, test_df, test_dataset):
    test_preds = trainer.predict(test_dataset)
    test_labels = np.argmax(test_preds.predictions, axis=1)

    monolingual = test_df[test_df["codemixed"] == 0]
    monolingual_gt = monolingual["offense"].values
    monolingual_preds = test_labels[monolingual.index]

    codemixed = test_df[test_df["codemixed"] == 1]
    codemixed_gt = codemixed["offense"].values
    codemixed_preds = test_labels[codemixed.index]

    results = {}
    results["monolingual"] = f1_score(monolingual_gt, monolingual_preds)
    results["codemixed"] = f1_score(codemixed_gt, codemixed_preds)
    results["overall"] = f1_score(test_df["offense"].values, test_labels)
    return results

def export_results(results, run_num):
    with open(f"data/predictions/{model_name}_run{run_num}.pickle", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    train_dataset, val_dataset, test_dataset = get_datasets(tokenizer, train_df, val_df, test_df)
    class_weights = get_class_weights(train_df)

    for run_num in range(1, NUM_RUNS+1):
        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
        trainer = get_trainer(model, train_dataset, val_dataset, class_weights)
        trainer.train()
        
        results = get_test_results(trainer, test_df, test_dataset)
        export_results(results, run_num)


if __name__ == "__main__":
    main()