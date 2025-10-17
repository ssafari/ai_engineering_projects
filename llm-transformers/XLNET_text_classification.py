import pandas as pd
import numpy as np
from cleantext import clean
import re
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer, pipeline
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datasets 
import evaluate
import random


def preprocess_data(csv_file: str):
    data = pd.read_csv(csv_file)
    data['text_clean'] = data['text'].apply(lambda x: clean(x, no_emoji=True))
    data['text_clean'] = data['text_clean'].apply(lambda x: re.sub('@[^\s]+', '', x))
    data.head(20)

    data['label'].value_counts().plot(kind="bar")
    g = data.groupby('label')
    data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    data['label'].value_counts().plot(kind="bar")

    data['label_int'] = LabelEncoder().fit_transform(data['label'])
    train_split, test_split = train_test_split(data, train_size = 0.8)
    train_split, val_split = train_test_split(train_split, train_size = 0.9)

    print(len(train_split))
    print(len(test_split))
    print(len(val_split))

    train_df = pd.DataFrame({
        "label": train_split.label_int.values,
        "text": train_split.text_clean.values
    })

    test_df = pd.DataFrame({
        "label": test_split.label_int.values,
        "text": test_split.text_clean.values
    })

    train_df = datasets.Dataset.from_dict(train_df)
    test_df = datasets.Dataset.from_dict(test_df)

    dataset_dict = datasets.DatasetDict({"train":train_df, "test":test_df})
    print(dataset_dict)
    return dataset_dict


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding = "max_length", max_length = 128, truncation=True)


def create_embeddings():
    data = './opt/lib/emotions_data.csv'
    dataset_dict = preprocess_data(data)
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    print(tokenized_datasets)

    print(tokenized_datasets['train']['text'][0])
    print(tokenized_datasets['train']['input_ids'][0])

    print(tokenized_datasets['train']['token_type_ids'][0])
    print(tokenized_datasets['train']['attention_mask'][0])

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

    fine_tune_model(tokenizer, small_train_dataset, small_eval_dataset)


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def fine_tune_model(tokenizer, small_train_dataset, small_eval_dataset):
    NUM_LABELS = 4
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', 
                                                       num_labels=NUM_LABELS, 
                                                       id2label={0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'})
    metric = evaluate.load("accuracy")
    print(metric)
    training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", num_train_epochs=3)
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics)

    trainer.train()

    # Evaluate tuned model
    trainer.evaluate()
    model.save_pretrained("fine_tuned_model")
    fine_tuned_model = XLNetForSequenceClassification.from_pretrained("fine_tuned_model")
    clf = pipeline("text-classification", fine_tuned_model, tokenizer=tokenizer)
    # rand_int = random.randint(0, len(val_split))
    # print(val_split['text_clean'][rand_int])
    # answer = clf(val_split['text_clean'][rand_int], top_k=None)
    # print(answer)