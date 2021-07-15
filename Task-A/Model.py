import os
import nltk
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
stop = stopwords.words("english")
stop += ["yeah","uh","ok","eh","um","uhm","hm","ye","yea"]
import pandas as pd
import numpy as np

from datasets import Dataset
from datasets import load_dataset, load_metric

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils import get_files

if "Models" not in os.listdir():
  os.mkdir("Models")

train_path = "./Files/train/"
dev_path = "./Files/dev/"

max_input_length = 2000
max_target_length = 200

def refine(ctext):
  segments = []
  for t in ctext:
    text = ""
    for word in t.split(" "):
      if word in stop:
        continue
      elif word.endswith("-"):
        continue
      elif "<" in word:
        continue
      elif word == "" or word ==" ":
        continue
      text += word + " "
    segments.append(text)
  return segments

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


text_train , ctext_train , folders_train = get_files(train_path)
text_dev , ctext_dev , folders_dev = get_files(dev_path)

segments_train = refine(ctext_train)
segments_dev = refine(ctext_dev)

id_train = list(range(len(ctext_train)))
id_dev = list(range(len(ctext_dev)))

train_df = pd.DataFrame(id_train,columns=["id"])
train_df["document"] = segments_train
train_df["summary"] = text_train
train_df["Folders"] = folders_train

val_df = pd.DataFrame(id_dev,columns=["id"])
val_df["document"] = segments_dev
val_df["summary"] = text_dev
val_df["Folders"] = folders_dev

train_df = Dataset.from_pandas(train_df)
val_df = Dataset.from_pandas(val_df)


model_checkpoint = "t5-base"
metric = load_metric("rouge")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_train = train_df.map(preprocess_function, batched=True)
tokenized_val = val_df.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to("cuda")

batch_size = 1
args = Seq2SeqTrainingArguments(
    "test-summarization",
    evaluation_strategy = "epoch",
    learning_rate=2e-8,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=6,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("training Started")
trainer.train()
trainer.save_model("./Models/Fine_Tuned")
