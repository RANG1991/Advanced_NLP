from transformers import AutoConfig, AutoModelForSequenceClassification
from prepare import prepare
from datasets import load_dataset

models_names = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]


def prepare_dataset(dataset_name):
    dataset = load_dataset(dataset_name)


def fine_tune_model(model_name, dataset_loader):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)


def main():
    prepare_dataset("sst2")
    for model_name in models_names:
        fine_tune_model(model_name)


if __name__ == "__main__":
    main()
