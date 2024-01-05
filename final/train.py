import argparse
import os

import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten() ** 2).tolist()

    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)

    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


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

y_train = y_train.map({"negative": 0, "neutral": 1, "positive": 2})
y_val = y_val.map({"negative": 0, "neutral": 1, "positive": 2})
y_test = y_test.map({"negative": 0, "neutral": 1, "positive": 2})


class SemevalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


argparser = argparse.ArgumentParser()
argparser.add_argument("--base_model", default="bert-base-uncased", type=str, help="Base model to finetune")
argparser.add_argument("--max_length", default=256, type=int, help="Max length of tokens")
argparser.add_argument("--lr", default=5e-5, type=float, help="Max length of tokens")
argparser.add_argument("--epochs", default=10, type=float, help="Number of training epochs")
args = argparser.parse_args()

LEARNING_RATE = args.lr
BASE_MODEL = args.base_model
MAX_LENGTH = args.max_length
EPOCHS = args.epochs
BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

train_encodings = tokenizer(X_train.values.tolist(), truncation=True, padding="max_length", max_length=MAX_LENGTH)
val_encodings = tokenizer(X_val.values.tolist(), truncation=True, padding="max_length", max_length=MAX_LENGTH)
test_encodings = tokenizer(X_test.values.tolist(), truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_dataset = SemevalDataset(train_encodings, y_train.values.tolist())
val_dataset = SemevalDataset(val_encodings, y_val.values.tolist())
test_dataset = SemevalDataset(test_encodings, y_test.values.tolist())

model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

training_args = TrainingArguments(
    output_dir=f"./models/{BASE_MODEL}-ml{MAX_LENGTH}-lr{LEARNING_RATE}-e{EPOCHS}",
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    save_total_limit=1,
)


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs[0][:, 0].float()
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_for_regression,
)

trainer.train()

test_pred = trainer.predict(test_dataset=test_dataset)
print(test_pred.metrics)
