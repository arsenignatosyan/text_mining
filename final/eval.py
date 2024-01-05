import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def json_save(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, help="Model to load")
argparser.add_argument("--max_length", default=256, type=int, help="Max length of tokens")
argparser.add_argument("--batch_size", default=16, type=int, help="Batch size")
argparser.add_argument("--save_path", default="./reports", type=str, help="Path to save reports to")
args = argparser.parse_args()

base_model_names = ["distilbert-base-uncased", "bert-base-uncased", "cardiffnlp/twitter-roberta-base-sentiment-latest"]
MODEL_PATH = args.model_path
for base_model_name in base_model_names:
    if base_model_name in MODEL_PATH:
        BASE_MODEL = base_model_name
        break
MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size
SAVE_PATH = args.save_path

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if MODEL_PATH[-1] == "/":
    MODEL_PATH = MODEL_PATH[:-1]

report_save_name = MODEL_PATH.split("/")[-2]
REPORT_SAVE_PATH = os.path.join(SAVE_PATH, report_save_name)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

base_path = "semeval-2017-tweets_Subtask-A/downloaded/"
base_df = pd.DataFrame()
colnames = ['id', 'sentiment', 'tweet']
for df_path in os.listdir(base_path):
    path = os.path.join(base_path, df_path)

    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] > 3:
        df = df.iloc[:, 0:3]
    df.columns = colnames
    base_df = pd.concat([base_df, df], ignore_index=True)

X = base_df["tweet"]
y = base_df["sentiment"]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42,
                                                            shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val,
                                                  random_state=42, shuffle=True)

y_test = y_test.map({"negative": 0, "neutral": 1, "positive": 2})

X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

nb_batches = math.ceil(len(X_test) / BATCH_SIZE)
y_preds = []

for i in range(nb_batches):
    input_texts = X_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    input_labels = y_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    encoded = tokenizer(input_texts.tolist(), truncation=True, padding="max_length", max_length=MAX_LENGTH,
                        return_tensors="pt")
    y_preds += model(**encoded).logits.reshape(-1).tolist()

df = pd.DataFrame({"text": X_test, "label": y_test, "prediction": y_preds})
df["rounded_prediction"] = df["prediction"].apply(round)
acc = accuracy_score(df["label"], df["rounded_prediction"])
_, rec, _, _ = precision_recall_fscore_support(df["label"], df["rounded_prediction"], average=None)
f1 = f1_score(df["label"], df["rounded_prediction"], average=None)

avg_rec = np.mean(rec)
avg_f1 = 0.5 * (f1[0] + f1[2])

dict_to_save = {
    "average_recall": avg_rec,
    "f1_pn": avg_f1,
    "accuracy": acc
}
print(dict_to_save)
json_save(data=dict_to_save, path=REPORT_SAVE_PATH)
