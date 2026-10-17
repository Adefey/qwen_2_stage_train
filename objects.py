import re

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTConfig
from trl.chat_template_utils import get_training_chat_template

from config import (
    DATASET_CUSTOM_EVAL_ITEMS,
    DATASET_GENERAL_EVAL_ITEMS_PER_SOURCE,
    DATASET_GENERAL_TOKENS_PER_CUSTOM_TOKEN,
    DATASET_INSTRUCTION_EVAL_ITEMS,
    DATASET_INSTRUCTION_TRAIN,
    DATASET_PRE_TRAIN,
    DATASETS_GENERAL_PRE_TRAIN,
    DIRECTORY_INSTRUCTION_TRAIN,
    DIRECTORY_PRE_TRAIN,
    EPOCHS_INSTRUCTION_TRAIN,
    EVAL_STEPS,
    EVAL_STEPS_INSTRUCTION_TRAIN,
    LEARNING_RATE_INSTRUCTION_TRAIN,
    LEARNING_RATE_PRE_TRAIN,
    LOGGING_STEPS,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    REPEATS_CHAT_PRE_TRAIN,
    REPEATS_FACTS_PRE_TRAIN,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = True


tokenizer_pre_train = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer_pre_train.pad_token is None:
    tokenizer_pre_train.pad_token = tokenizer_pre_train.eos_token


# Official Qwen3.5 template; TRL swaps in its training variant with {% generation %} markers
tokenizer_instruction_train = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer_instruction_train.pad_token is None:
    tokenizer_instruction_train.pad_token = tokenizer_instruction_train.eos_token
# Raises here, before hours of PT, if TRL cannot patch this template for assistant_only_loss
get_training_chat_template(tokenizer_instruction_train)


def tokenize_function_pre_train(examples: dict):
    return tokenizer_pre_train(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )


def clean_text(text: str) -> str:
    text = text.replace("\u0301", "")
    text = re.sub(r"\(\s*[,;\s]*\)", "", text)
    text = re.sub(r"\(([^()]*?)(?:[;,]\s*)+\)", r"(\1)", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def take_general(name: str, subset: str, token_budget: int, eval_items: int):
    stream = load_dataset(name, subset, split="train", streaming=True).shuffle(
        seed=14565, buffer_size=10_000
    )
    train_items, eval_list, used_tokens = [], [], 0
    for doc in stream:
        text = clean_text(doc["text"][: MAX_SEQ_LENGTH * 8])
        if len(eval_list) < eval_items:
            eval_list.append({"text": text})
            continue
        train_items.append({"text": text})
        used_tokens += min(len(tokenizer_pre_train(text)["input_ids"]), MAX_SEQ_LENGTH)
        if used_tokens >= token_budget:
            break
    return Dataset.from_list(train_items), Dataset.from_list(eval_list)


split_custom_pre_train = Dataset.from_json(path_or_paths=DATASET_PRE_TRAIN).train_test_split(
    test_size=DATASET_CUSTOM_EVAL_ITEMS, seed=14565
)
dataset_custom_pre_train = split_custom_pre_train["train"]
dataset_custom_eval_pre_train = split_custom_pre_train["test"]


split_instruction_train = Dataset.from_json(
    path_or_paths=DATASET_INSTRUCTION_TRAIN
).train_test_split(test_size=DATASET_INSTRUCTION_EVAL_ITEMS, seed=14565)
raw_dataset_instruction_train = split_instruction_train["train"]
raw_dataset_instruction_eval = split_instruction_train["test"]


# Dialogues as plain text keep the chat format alive while PT pushes the model towards raw text
dataset_chat_pre_train = Dataset.from_list([
    {"text": tokenizer_instruction_train.apply_chat_template(item["messages"], tokenize=False)}
    for item in raw_dataset_instruction_train
])


dataset_custom_all_pre_train = concatenate_datasets(
    [dataset_custom_pre_train] * REPEATS_FACTS_PRE_TRAIN
    + [dataset_chat_pre_train] * REPEATS_CHAT_PRE_TRAIN
)
custom_tokens = sum(
    len(ids) for ids in tokenize_function_pre_train(dataset_custom_all_pre_train[:])["input_ids"]
)
general_tokens_per_source = int(
    custom_tokens * DATASET_GENERAL_TOKENS_PER_CUSTOM_TOKEN / len(DATASETS_GENERAL_PRE_TRAIN)
)


general_train_parts, general_eval_parts = [], []
for general_name, general_subset in DATASETS_GENERAL_PRE_TRAIN:
    general_train, general_eval = take_general(
        general_name,
        general_subset,
        general_tokens_per_source,
        DATASET_GENERAL_EVAL_ITEMS_PER_SOURCE,
    )
    general_train_parts.append(general_train)
    general_eval_parts.append(general_eval)


dataset_pre_train = concatenate_datasets(
    [dataset_custom_all_pre_train, *general_train_parts]
).shuffle(seed=14565)


tokenized_dataset_pre_train = dataset_pre_train.map(
    tokenize_function_pre_train,
    batched=True,
    remove_columns=["text"],
)


tokenized_eval_datasets_pre_train = {
    "facts": dataset_custom_eval_pre_train.map(
        tokenize_function_pre_train, batched=True, remove_columns=["text"]
    ),
    "general": (
        concatenate_datasets(general_eval_parts).map(
            tokenize_function_pre_train, batched=True, remove_columns=["text"]
        )
    ),
}


training_args_pre_train = TrainingArguments(
    output_dir=DIRECTORY_PRE_TRAIN,
    num_train_epochs=1,
    learning_rate=LEARNING_RATE_PRE_TRAIN,
    lr_scheduler_type="cosine",
    warmup_steps=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    eval_on_start=True,
    per_device_eval_batch_size=2,
    save_strategy="no",
    report_to="wandb",
    bf16=use_bf16,
    fp16=False,
    use_cpu=(device == "cpu"),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    train_sampling_strategy="group_by_length",
)


sft_config = SFTConfig(
    output_dir=DIRECTORY_INSTRUCTION_TRAIN,
    num_train_epochs=EPOCHS_INSTRUCTION_TRAIN,
    learning_rate=LEARNING_RATE_INSTRUCTION_TRAIN,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS_INSTRUCTION_TRAIN,
    eval_on_start=True,
    per_device_eval_batch_size=2,
    save_strategy="no",
    assistant_only_loss=True,
    max_length=MAX_SEQ_LENGTH,
    report_to="wandb",
    bf16=use_bf16,
    fp16=False,
    use_cpu=(device == "cpu"),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
)
