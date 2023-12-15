from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd
import numpy as np
import os 
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from matplotlib import pyplot as plt

import wandb, random

wandb.init(
    # set the wandb project where this run will be logged
    project="Llama Linear Probe - Expt 1",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 5e-4,
    "weightDecay": 1e-3,
    "architecture": "LLaMAForSeqClassification - Llama7b",
    "dataset": "CodeMixed Dataset",
    "epochs": 10,
    "batch_size": 64
    }
)


token="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"
n_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto", 
    num_labels=n_labels,
    token=token
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token, padding="max_length", truncation=True)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False 


for n, p in model.named_parameters():
    if n != 'score.weight':
        p.requires_grad = False

for n, p in model.named_parameters():
    print(f"{n}: Requires Grad: {p.requires_grad}")

    # Dataset Class
class CodeMixedDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        data = pd.read_csv(data_path)[["tweet_text", "offense"]]
        self.text = data["tweet_text"].tolist()
        self.labels = torch.tensor(data["offense"])
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, ix):
        text_inputs = tokenizer(self.text[ix], padding="max_length", truncation=True, return_tensors="pt", max_length=128)
        labels = self.labels[ix]
        
        return {
            "text": text_inputs,
            "labels": labels
        }

batch_size = 64

train_dataset = CodeMixedDataset(data_path="data/train.csv", tokenizer=tokenizer)
val_dataset = CodeMixedDataset(data_path="data/val.csv", tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, dataloader, optimizer, loss_fn, batch_size, epoch):
    loss_total = 0
    epoch_preds = []
    epoch_labels = []
    model.train()
    with tqdm(total=len(dataloader), position=0, leave=True) as pbar:
        for i, batch in tqdm(enumerate(dataloader), position=0, leave=True):
            pbar.update()
            input_ids = batch["text"]["input_ids"].squeeze(1).to("cuda")
            attention_mask = batch["text"]["attention_mask"].squeeze(1).to("cuda")
            labels = batch["labels"].to("cuda")
            logits = model(input_ids, attention_mask).logits
            epoch_preds += torch.argmax(logits, dim=1).tolist()
            epoch_labels += labels.tolist()
            loss = loss_fn(logits, labels)
            wandb.log({
                "train_step": (epoch*len(dataloader)) + (i+1),
                "train_loss": loss.item()
            })
            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
    epoch_precision = precision_score(epoch_labels, epoch_preds)
    epoch_recall = recall_score(epoch_labels, epoch_preds)
    epoch_f1 = f1_score(epoch_labels, epoch_preds)
   
    results = {
        "epoch": epoch+1,
        "train_accuracy": epoch_accuracy,
        "train_precision": epoch_precision,
        "train_recall": epoch_recall,
        "train_f1": epoch_f1
    }
    wandb.log(results)
    
    print(f"Epoch {epoch+1}: Train Per Sample Loss = {loss_total/len(dataloader)/batch_size}")
    print(f"Epoch {i+1}: Accuracy={epoch_accuracy}, Precision={epoch_precision}, Recall={epoch_recall}, F1={epoch_f1}")

    return results

def val_epoch(model, dataloader, loss_fn, batch_size, i):
    loss_total = 0
    epoch_preds = []
    epoch_labels = []
    model.eval()
    with tqdm(total=len(dataloader), position=0, leave=True) as pbar:
        for batch in tqdm(dataloader, position=0, leave=True):
            pbar.update()
            input_ids = batch["text"]["input_ids"].squeeze(1).to("cuda")
            attention_mask = batch["text"]["attention_mask"].squeeze(1).to("cuda")
            labels = batch["labels"].to("cuda")
            logits = model(input_ids, attention_mask).logits
            epoch_preds += torch.argmax(logits, dim=1).tolist()
            epoch_labels += labels.tolist()
            loss = loss_fn(logits, labels)
            loss_total += loss.item()

    epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
    epoch_precision = precision_score(epoch_labels, epoch_preds)
    epoch_recall = recall_score(epoch_labels, epoch_preds)
    epoch_f1 = f1_score(epoch_labels, epoch_preds)

    results = {
        "epoch": i,
        "val_accuracy": epoch_accuracy,
        "val_precision": epoch_precision,
        "val_recall": epoch_recall,
        "val_f1": epoch_f1
    }
    
    wandb.log(results)
    
    
    print(f"Epoch {i+1}: Validation Per Sample Loss = {loss_total/len(dataloader)/batch_size}")
    print(f"Epoch {i+1}: Accuracy={epoch_accuracy}, Precision={epoch_precision}, Recall={epoch_recall}, F1={epoch_f1}")
    
    return results

test_dataset = CodeMixedDataset(data_path="data/test.csv", tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def test_epoch(model, best_model_ckpt, dataloader, loss_fn, batch_size, i):
    ckpt = torch.load(best_model_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Testing model corresponding to Epoch: {ckpt['epoch']}")
    loss_total = 0
    epoch_preds = []
    epoch_labels = []
    model.eval()
    with tqdm(total=len(dataloader), position=0, leave=True) as pbar:
        for batch in tqdm(dataloader, position=0, leave=True):
            pbar.update()
            input_ids = batch["text"]["input_ids"].squeeze(1).to("cuda")
            attention_mask = batch["text"]["attention_mask"].squeeze(1).to("cuda")
            labels = batch["labels"].to("cuda")
            logits = model(input_ids, attention_mask).logits
            epoch_preds += torch.argmax(logits, dim=1).tolist()
            epoch_labels += labels.tolist()
            loss = loss_fn(logits, labels)
            loss_total += loss.item()

    epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
    epoch_precision = precision_score(epoch_labels, epoch_preds)
    epoch_recall = recall_score(epoch_labels, epoch_preds)
    epoch_f1 = f1_score(epoch_labels, epoch_preds)
    
    print(f"Epoch {i}: Test Per Sample Loss = {loss_total/len(dataloader)/batch_size}")
    print(f"Epoch {i}: Accuracy={epoch_accuracy}, Precision={epoch_precision}, Recall={epoch_recall}, F1={epoch_f1}")
    
#Loss and Optimizer
learning_rate = 5e-4
weight_decay = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#Training epochs
num_epochs = 10

train_loss_ovr = []
val_loss_ovr = []

best_val_f1 = 0.0
with tqdm(total=num_epochs, position=0, leave=True) as pbar:
    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        train_epoch_results = train_epoch(model=model, 
                                       dataloader=train_dataloader, 
                                       optimizer=optimizer, 
                                       loss_fn=loss_fn, 
                                       batch_size=batch_size, 
                                       epoch=epoch)
        
        val_epoch_results = val_epoch(model=model, 
                                    dataloader=val_dataloader, 
                                    loss_fn=loss_fn, 
                                    batch_size=batch_size, 
                                    i=epoch)
        val_f1 = val_epoch_results["val_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, "linear_probing_best_ckpt.pth")
        
test_epoch(model, "linear_probing_best_ckpt.pth", test_dataloader, loss_fn, batch_size, epoch)
        
