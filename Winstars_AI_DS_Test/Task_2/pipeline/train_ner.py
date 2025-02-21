import torch
import json
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

#  Load the synthetic dataset
with open("synthetic_ner_data.json", "r") as f:
    data = json.load(f)

#  Convert to Hugging Face Dataset format
dataset = Dataset.from_list(data)

#  Split dataset into train (80%) and validation (20%)
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

#  Load pre-trained NER model
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

#  Tokenize dataset
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"], 
        truncation=True, 
        padding="max_length",  #  Add padding to ensure uniform length
        max_length=128,  #  Adjust max token length (you can change this)
        is_split_into_words=True
    )
    
    labels = []
    word_ids = tokenized_inputs.word_ids()
    
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # Ignore padding tokens
        else:
            labels.append(example["ner_tags"][word_idx])
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels)
val_dataset = val_dataset.map(tokenize_and_align_labels)

#  Define training arguments (Enable Evaluation)
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",  #Enable validation dataset
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_total_limit=2,
    load_best_model_at_end=True
)

#  Trainer setup (Add `eval_dataset`)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  
    tokenizer=tokenizer
)

#  Train the model
trainer.train()

#  Save trained model
trainer.save_model("./ner_model")
tokenizer.save_pretrained("./ner_model")
print(" NER model training completed and saved to './ner_model'")
