from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from prepare import prepare
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn

models_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_dataset(dataset_name, batch_size):
    dataset = load_dataset(dataset_name)
    dataloader_train = DataLoader(dataset["train"], batch_size=batch_size)
    dataloader_val = DataLoader(dataset["validation"], batch_size=batch_size)
    dataloader_test = DataLoader(dataset["test"], batch_size=batch_size)
    return dataloader_train, dataloader_val, dataloader_test


def train_epoch(model, tokenizer, data_loader, optimizer, loss_func):
    model.train()
    running_loss = 0.0
    for X, y in data_loader:
        X = tokenizer.encode(X, max_length=tokenizer.model_max_length, padding=True, truncation=True,
                             return_tensors='pt')
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        optimizer.zero_grad()
        loss = loss_func(y, y_hat.squeeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Loss: {loss.item():.4f}")
    print(f"Loss on the entire training epoch: {running_loss / (len(data_loader)):.4f}")
    return running_loss / (len(data_loader))


def prepare_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer(model_name)
    return model, tokenizer, config


def main():
    prepare()
    dataloader_train, dataloader_val, dataloader_test = prepare_dataset("sst2", 32)
    loss_func = nn.CrossEntropyLoss()
    for model_name in models_names:
        model, tokenizer, config = prepare_model(model_name)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
        for epoch in config.num_epochs:
            train_epoch(model, tokenizer, dataloader_train, loss_func, optimizer)


if __name__ == "__main__":
    main()
