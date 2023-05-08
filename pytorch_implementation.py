from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch
from torch import nn
import random
import numpy as np

models_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
device = ("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, tokenizer, data_loader, loss_func, optimizer):
    model.train()
    running_loss = 0
    num_examples = 0
    for dict_example in data_loader:
        X = dict_example["sentence"]
        y = dict_example["label"]
        X = tokenizer(X, max_length=tokenizer.model_max_length, padding=True, truncation=True,
                      return_tensors='pt')
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        optimizer.zero_grad()
        loss = loss_func(y_hat, y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_examples += 1
        if num_examples % 100 == 0:
            print(f"num examples so far: {num_examples}")
    print(f"Loss on the entire training epoch: {running_loss / (len(data_loader)):.4f}")
    return running_loss / (len(data_loader))


def val_epoch(model, tokenizer, data_loader):
    model.eval()
    running_acc = 0
    num_examples = 0
    with torch.no_grad():
        for dict_example in data_loader:
            X = dict_example["sentence"]
            y = dict_example["label"]
            X = tokenizer(X, max_length=tokenizer.model_max_length, padding=True, truncation=True,
                          return_tensors='pt')
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            acc_batch = torch.mean((y_hat.round().float() == y.float()).float())
            running_acc += acc_batch
            num_examples += 1
            if num_examples % 100 == 0:
                print(f"accuracy so far: {running_acc / num_examples}")
        print(f"accuracy on the entire validation epoch: {running_acc / (len(data_loader)):.4f}")
    return running_acc / (len(data_loader))


class Pre_Train_On_SST2(nn.Module):
    def __init__(self, model_name):
        super(Pre_Train_On_SST2, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = 2
        self.model_seq_cls = AutoModelForSequenceClassification.from_config(config=self.config)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        outputs = self.model_seq_cls(**X)
        output = self.softmax(outputs.logits)[:, 0]
        return output


def prepare_model_pytorch(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Pre_Train_On_SST2(model_name)
    model = model.to(device)
    return model, tokenizer


def train_and_validate_using_pytorch(args, dataset_train, dataset_val, model, tokenizer):
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    list_acc = []
    dataloader_train = DataLoader(dataset_train, batch_size=args.per_device_train_batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.per_device_train_batch_size, shuffle=False)
    for _ in range(int(args.num_train_epochs)):
        train_epoch(model, tokenizer, dataloader_train, loss_func, optimizer)
        acc_epoch = val_epoch(model, tokenizer, dataloader_val)
        list_acc.append(acc_epoch.item())
    return list_acc


def predict_using_pytorch(model_name, dict_model_name_to_model_obj_and_best_acc_seed, test_dataset):
    best_model_obj, best_tokenizer_obj, best_seed = dict_model_name_to_model_obj_and_best_acc_seed[model_name]
    initialize_seed(best_seed)
    best_model_obj.eval()
    list_predictions = []
    with torch.no_grad():
        for dict_example in test_dataset:
            X = dict_example["sentence"][:]
            X = best_tokenizer_obj(X, max_length=best_tokenizer_obj.model_max_length,
                                   truncation=True, return_tensors='pt')
            X = X.to(device)
            y_hat = best_model_obj(X)
            list_predictions.append((dict_example["sentence"], y_hat.round().item()))
    return list_predictions


def initialize_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
