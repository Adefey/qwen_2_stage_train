import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTConfig

from config import (
    CHAT_TEMPLATE,
    DATASET_GENERAL_PRE_TRAIN,
    DATASET_GENERAL_SUBSET_PRE_TRAIN,
    DATASET_INSTRUCTION_TRAIN,
    DATASET_PRE_TRAIN,
    DIRECTORY_INSTRUCTION_TRAIN,
    DIRECTORY_PRE_TRAIN,
    EPOCHS_INSTRUCTION_TRAIN,
    EPOCHS_PRE_TRAIN,
    LEARNING_RATE_INSTRUCTION_TRAIN,
    LEARNING_RATE_PRE_TRAIN,
    LOGGING_STEPS,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = True

tokenizer_pre_train = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer_pre_train.pad_token is None:
    tokenizer_pre_train.pad_token = tokenizer_pre_train.eos_token

tokenizer_instruction_train = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_instruction_train.chat_template = CHAT_TEMPLATE
if tokenizer_instruction_train.pad_token is None:
    tokenizer_instruction_train.pad_token = tokenizer_instruction_train.eos_token


def tokenize_function_pre_train(examples: dict):
    return tokenizer_pre_train(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )


dataset_custom_pre_train = Dataset.from_json(path_or_paths=DATASET_PRE_TRAIN)

stream_general_pre_train = load_dataset(
    DATASET_GENERAL_PRE_TRAIN,
    DATASET_GENERAL_SUBSET_PRE_TRAIN,
    split="train",
    streaming=True,
)

num_general_samples = len(dataset_custom_pre_train) * 4

general_samples = list(stream_general_pre_train.select_columns(["text"]).take(num_general_samples))
dataset_general_pre_train = Dataset.from_list(general_samples)

dataset_pre_train = concatenate_datasets(
    [dataset_custom_pre_train, dataset_general_pre_train]
).shuffle(seed=14565)

tokenized_dataset_pre_train = dataset_pre_train.map(
    tokenize_function_pre_train,
    batched=True,
    remove_columns=["text"],
)

training_args_pre_train = TrainingArguments(
    output_dir=DIRECTORY_PRE_TRAIN,
    num_train_epochs=EPOCHS_PRE_TRAIN,
    learning_rate=LEARNING_RATE_PRE_TRAIN,
    lr_scheduler_type="cosine",
    warmup_steps=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",
    bf16=use_bf16,
    fp16=False,
    use_cpu=(device == "cpu"),
)

raw_dataset_instruction_train = Dataset.from_json(path_or_paths=DATASET_INSTRUCTION_TRAIN)

sft_config = SFTConfig(
    output_dir=DIRECTORY_INSTRUCTION_TRAIN,
    num_train_epochs=EPOCHS_INSTRUCTION_TRAIN,
    learning_rate=LEARNING_RATE_INSTRUCTION_TRAIN,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    assistant_only_loss=True,
    max_length=MAX_SEQ_LENGTH,
    report_to="wandb",
    bf16=use_bf16,
    fp16=False,
    use_cpu=(device == "cpu"),
)
