import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
import evaluate
import random
import argparse
from time import time
# import wandb

# use_wandb = False
models_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
device = ("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(num_samples_train, num_samples_validation, num_samples_test):
    dataset = load_dataset("glue", "sst2")
    if num_samples_train != -1:
        dataset_train = dataset["train"].select(range(num_samples_train))
    else:
        dataset_train = dataset["train"]
    if num_samples_validation != -1:
        dataset_val = dataset["validation"].select(range(num_samples_validation))
    else:
        dataset_val = dataset["validation"]
    if num_samples_test != -1:
        dataset_test = dataset["test"].select(range(num_samples_test))
    else:
        dataset_test = dataset["test"]
    return dataset_train, dataset_val, dataset_test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def prepare_model_hugging_face(model_name, args, dataset_train, dataset_val, seed):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    args.evaluation_strategy = "epoch"
    args.save_strategy = "no"
    # if use_wandb:
    #     args.report_to = ["wandb"]
    #     args.run_name = model_name
    # else:
    args.report_to = []
    dataset_train = dataset_train.map(lambda example: tokenizer(example["sentence"],
                                                                max_length=tokenizer.model_max_length,
                                                                truncation=True), batched=True).shuffle(seed=seed)
    dataset_val = dataset_val.map(lambda example: tokenizer(example["sentence"],
                                                            max_length=tokenizer.model_max_length,
                                                            truncation=True), batched=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)
    return trainer, tokenizer


def train_and_validate_using_hugging_face(trainer):
    trainer.train()
    metrics = trainer.evaluate()
    return [metrics["eval_accuracy"]]


def change_label_to_zero(example):
    example["label"] = [0] * len(example["label"])
    return example


def predict_using_hugging_face(model_name, dict_model_name_to_model_obj_and_best_acc_seed, test_dataset):
    best_trainer, best_tokenizer, best_seed = dict_model_name_to_model_obj_and_best_acc_seed[model_name]
    initialize_seed(best_seed)
    test_dataset = test_dataset.map(lambda example: best_tokenizer(example["sentence"]))
    test_dataset = test_dataset.map(change_label_to_zero, batched=True)
    predictions = best_trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    examples_with_preds = zip([example["sentence"] for example in test_dataset], preds)
    return examples_with_preds


def create_res_file(dict_model_name_to_acc_list, training_time, prediction_time):
    with open("res.txt", "w") as f:
        for model_name in dict_model_name_to_acc_list.keys():
            all_acc_list = dict_model_name_to_acc_list[model_name]
            f.write(f"{model_name},{np.mean(all_acc_list)} +- {np.std(all_acc_list)}\n")
        f.write("----\n")
        f.write(f"train time,{training_time}\n")
        f.write(f"predict time,{prediction_time}")


def create_predictions_file(list_predictions):
    with open("predictions.txt", "w") as f:
        for sentence, prediction in list_predictions:
            f.write(f"{sentence}###{prediction}\n")


def initialize_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(prog='fine-tuning models',
                                     description='fine-tune three selected models on SST2 dataset')
    parser.add_argument('num_seeds', type=int)
    parser.add_argument('num_samples_train', type=int)
    parser.add_argument('num_samples_validation', type=int)
    parser.add_argument('num_samples_test', type=int)
    command_args = parser.parse_args()
    dataset_train, dataset_val, dataset_test = prepare_dataset(command_args.num_samples_train,
                                                               command_args.num_samples_validation,
                                                               command_args.num_samples_test)
    dict_model_name_to_acc_list = {}
    dict_model_name_to_model_obj_and_best_acc_seed = {}
    start_training_time = time()
    for model_name in models_names:
        best_mean_seed_acc = None
        best_seed = None
        best_trainer_obj = None
        best_tokenizer_obj = None
        for seed in range(command_args.num_seeds):
            # initialize_seed(seed)
            transformers.set_seed(seed)
            training_args = TrainingArguments("working_dir")
            trainer, tokenizer = prepare_model_hugging_face(model_name, training_args, dataset_train, dataset_val, seed)
            list_acc = train_and_validate_using_hugging_face(trainer)
            if model_name not in dict_model_name_to_acc_list.keys():
                dict_model_name_to_acc_list[model_name] = []
            dict_model_name_to_acc_list[model_name].extend(list_acc)
            if best_mean_seed_acc is None or best_mean_seed_acc < np.mean(list_acc):
                best_mean_seed_acc = np.mean(list_acc)
                best_seed = seed
                best_trainer_obj = trainer
                best_tokenizer_obj = tokenizer
        dict_model_name_to_model_obj_and_best_acc_seed[model_name] = (best_trainer_obj, best_tokenizer_obj, best_seed)
        # if use_wandb:
        #     wandb.finish()
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
    list_predictions = predict_using_hugging_face(model_name_with_max_acc,
                                                  dict_model_name_to_model_obj_and_best_acc_seed,
                                                  dataset_test)
    dur_prediction_time = time() - start_prediction_time
    create_res_file(dict_model_name_to_acc_list, dur_training_time, dur_prediction_time)
    create_predictions_file(list_predictions)


if __name__ == "__main__":
    main()
