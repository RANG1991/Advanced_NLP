from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from prepare import prepare
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import evaluate

models_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
seed = 123
device = ("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    dataset_train = dataset["train"]
    dataset_val = dataset["validation"]
    dataset_test = dataset["test"]
    return dataset_train, dataset_val, dataset_test


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
        print(f"Loss on the entire training epoch: {running_acc / (len(data_loader)):.4f}")
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


def prepare_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Pre_Train_On_SST2(model_name)
    model = model.to(device)
    return model, tokenizer


def train_and_validate_using_pytorch(args, dataset_train, dataset_val, model_name):
    loss_func = nn.BCELoss()
    model, tokenizer = prepare_model(model_name)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    for _ in range(int(args.num_train_epochs)):
        dataloader_train = DataLoader(dataset_train, batch_size=args.per_device_train_batch_size, shuffle=True)
        train_epoch(model, tokenizer, dataloader_train, loss_func, optimizer)
    # for _ in range(int(args.num_val_epochs)):
        dataloader_val = DataLoader(dataset_val, batch_size=args.per_device_train_batch_size, shuffle=False)
        val_epoch(model, tokenizer, dataloader_val)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def train_and_validate_using_hugging_face(args, dataset_train, dataset_val, model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_config(config=config)
    args.evaluation_strategy = "epoch"
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train.shuffle(seed=seed),
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics)
    trainer.train()


def main():
    prepare()
    args = TrainingArguments("working_dir")
    dataset_train, dataset_val, dataset_test = prepare_dataset("sst2")
    for model_name in models_names:
        train_and_validate_using_pytorch(args, dataset_train, dataset_val, model_name)
    for model_name in models_names:
        train_and_validate_using_hugging_face(args, dataset_train, dataset_val, model_name)


if __name__ == "__main__":
    main()
