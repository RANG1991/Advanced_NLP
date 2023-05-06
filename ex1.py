from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from prepare import prepare
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import TrainingArguments

models_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_dataset(dataset_name, batch_size):
    dataset = load_dataset(dataset_name)
    dataloader_train = DataLoader(dataset["train"], batch_size=batch_size)
    dataloader_val = DataLoader(dataset["validation"], batch_size=batch_size)
    dataloader_test = DataLoader(dataset["test"], batch_size=batch_size)
    return dataloader_train, dataloader_val, dataloader_test


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


def main():
    prepare()
    args = TrainingArguments("working_dir")
    dataloader_train, dataloader_val, dataloader_test = prepare_dataset("sst2", args.per_device_train_batch_size)
    loss_func = nn.BCELoss()
    for model_name in models_names:
        model, tokenizer = prepare_model(model_name)
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        for epoch in range(int(args.num_train_epochs)):
            train_epoch(model, tokenizer, dataloader_train, loss_func, optimizer)


if __name__ == "__main__":
    main()
