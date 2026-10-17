import os

import torch

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-0.8B-Base")

DATA_TYPE = torch.bfloat16

LEARNING_RATE_PRE_TRAIN = 2e-5
LEARNING_RATE_INSTRUCTION_TRAIN = 1e-5
MAX_SEQ_LENGTH = 1024

DATASET_PRE_TRAIN = "dataset_pt.jsonl"
DATASET_GENERAL_PRE_TRAIN = "HuggingFaceTB/smollm-corpus"
DATASET_GENERAL_SUBSET_PRE_TRAIN = "cosmopedia-v2"
DATASET_GENERAL_ITEMS_PER_CUSTOM_ITEM = 5
DATASET_INSTRUCTION_TRAIN = "dataset_it.jsonl"

DIRECTORY_PRE_TRAIN = "./pre_train_dir"
DIRECTORY_INSTRUCTION_TRAIN = "./it_train_dir"

EPOCHS_PRE_TRAIN = 2
EPOCHS_INSTRUCTION_TRAIN = 2

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% generation %}{{ message['content'] + '<|im_end|>\n' }}{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

LOGGING_STEPS = 5
