from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from prepare import prepare
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import evaluate
import random
import argparse
from time import time

models_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
device = ("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(dataset_name, num_samples_train, num_samples_validation, num_samples_test):
    dataset = load_dataset(dataset_name)
    dataset_train = dataset["train"].select(range(num_samples_train))
    dataset_val = dataset["validation"].select(range(num_samples_validation))
    dataset_test = dataset["test"].select(range(num_samples_test))
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


def prepare_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Pre_Train_On_SST2(model_name)
    model = model.to(device)
    return model, tokenizer


def train_and_validate_using_pytorch(args, dataset_train, dataset_val, model_name):
    loss_func = nn.BCELoss()
    model, tokenizer = prepare_model(model_name)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    list_acc = []
    for _ in range(int(args.num_train_epochs)):
        dataloader_train = DataLoader(dataset_train, batch_size=args.per_device_train_batch_size, shuffle=True)
        train_epoch(model, tokenizer, dataloader_train, loss_func, optimizer)
        # for _ in range(int(args.num_val_epochs)):
        dataloader_val = DataLoader(dataset_val, batch_size=args.per_device_train_batch_size, shuffle=False)
        acc_epoch = val_epoch(model, tokenizer, dataloader_val)
        list_acc.append(acc_epoch.item())
    return list_acc


def predict_using_pytorch(model_name, test_dataset):
    model, tokenizer = prepare_model(model_name)
    model.eval()
    list_predictions = []
    with torch.no_grad():
        for dict_example in test_dataset:
            X = dict_example["sentence"][:]
            X = tokenizer(X, max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
            X = X.to(device)
            y_hat = model(X)
            list_predictions.append((dict_example["sentence"], y_hat.round().item()))
    return list_predictions


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def train_and_validate_using_hugging_face(args, dataset_train, dataset_val, model_name, seed):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset_train = dataset_train.map(lambda example: tokenizer(example["sentence"],
                                                                max_length=tokenizer.model_max_length, padding=True,
                                                                truncation=True, return_tensors='pt'), batched=True)
    dataset_val = dataset_val.map(lambda example: tokenizer(example["sentence"],
                                                            max_length=tokenizer.model_max_length, padding=True,
                                                            truncation=True, return_tensors='pt'), batched=True)
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


def create_res_file(dict_model_name_to_acc_list, training_time, prediction_time):
    with open("./res_Ran.txt", "w") as f:
        for model_name in dict_model_name_to_acc_list.keys():
            all_acc_list = dict_model_name_to_acc_list[model_name]
            f.write(f"{model_name},{np.mean(all_acc_list)} +- {np.std(all_acc_list)}\n")
        f.write("----\n")
        f.write(f"train time,{training_time}\n")
        f.write(f"predict time,{prediction_time}")


def create_predictions_file(list_predictions):
    with open("./predictions_Ran.txt", "w") as f:
        for sentence, prediction in list_predictions:
            f.write(f"{sentence}###{prediction}\n")


def initialize_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    prepare()
    parser = argparse.ArgumentParser(prog='fine-tuning models',
                                     description='fine-tune three selected models on SST2 dataset')
    parser.add_argument('num_seeds', type=int)
    parser.add_argument('num_samples_train', type=int)
    parser.add_argument('num_samples_validation', type=int)
    parser.add_argument('num_samples_test', type=int)
    command_args = parser.parse_args()
    dataset_train, dataset_val, dataset_test = prepare_dataset("sst2",
                                                               command_args.num_samples_train,
                                                               command_args.num_samples_validation,
                                                               command_args.num_samples_test)
    dict_model_name_to_acc_list = {}
    dict_model_name_to_best_acc_seed = {}
    start_training_time = time()
    for model_name in models_names:
        best_mean_seed_acc = None
        best_seed = None
        for seed in range(command_args.num_seeds):
            initialize_seed(seed)
            training_args = TrainingArguments("working_dir")
            list_acc = train_and_validate_using_pytorch(training_args, dataset_train, dataset_val, model_name)
            if model_name not in dict_model_name_to_acc_list.keys():
                dict_model_name_to_acc_list[model_name] = []
            dict_model_name_to_acc_list[model_name].extend(list_acc)
            if best_mean_seed_acc is None or best_mean_seed_acc < np.mean(list_acc):
                best_mean_seed_acc = np.mean(list_acc)
                best_seed = seed
        dict_model_name_to_best_acc_seed[model_name] = best_seed
        # for model_name in models_names:
        #     train_and_validate_using_hugging_face(training_args, dataset_train, dataset_val, model_name, seed)
    dur_training_time = time() - start_training_time
    start_prediction_time = time()
    model_name_with_max_acc = None
    best_acc = None
    for model_name in models_names:
        all_acc_list = dict_model_name_to_acc_list[model_name]
        mean_acc = np.mean(all_acc_list)
        if best_acc is None or best_acc < mean_acc:
            model_name_with_max_acc = model_name
            best_acc = mean_acc
    list_predictions = predict_using_pytorch(model_name_with_max_acc, dataset_test)
    dur_prediction_time = time() - start_prediction_time
    create_res_file(dict_model_name_to_acc_list, dur_training_time, dur_prediction_time)
    create_predictions_file(list_predictions)


if __name__ == "__main__":
    main()
