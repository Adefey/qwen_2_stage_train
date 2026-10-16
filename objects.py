import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTConfig

from config import (
    CHAT_TEMPLATE,
    DATASET_INSTRUCTION_TRAIN,
    DATASET_PRE_TRAIN,
    DIRECTORY_INSTRUCTION_TRAIN,
    DIRECTORY_PRE_TRAIN,
    EPOCHS_INSTRUCTION_TRAIN,
    EPOCHS_PRE_TRAIN,
    LEARNING_RATE_INSTRUCTION_TRAIN,
    LEARNING_RATE_PRE_TRAIN,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    LOGGING_STEPS,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_pre_train = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_instruction_train = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_instruction_train.chat_template = CHAT_TEMPLATE


def tokenize_function_pre_train(examples: dict):
    return tokenizer_pre_train(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)


def tokenize_function_instruction_train(example: dict):
    return tokenizer_instruction_train.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
    )


tokenized_dataset_pre_train = Dataset.from_json(path_or_paths=DATASET_PRE_TRAIN).map(
    tokenize_function_pre_train, batched=True, remove_columns=["text"]
)
training_args_pre_train = TrainingArguments(
    output_dir=DIRECTORY_PRE_TRAIN,
    num_train_epochs=EPOCHS_PRE_TRAIN,
    learning_rate=LEARNING_RATE_PRE_TRAIN,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    report_to="none",
    bf16=False,
)

tokenized_dataset_instruction_train = Dataset.from_json(path_or_paths=DATASET_INSTRUCTION_TRAIN).map(
    tokenize_function_instruction_train, remove_columns=["messages"]
)
sft_config = SFTConfig(
    output_dir=DIRECTORY_INSTRUCTION_TRAIN,
    num_train_epochs=EPOCHS_INSTRUCTION_TRAIN,
    learning_rate=LEARNING_RATE_INSTRUCTION_TRAIN,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    assistant_only_loss=True,
    max_length=MAX_SEQ_LENGTH,
    report_to="none",
    bf16=False,
)
