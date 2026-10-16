import torch

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
DATA_TYPE = torch.float16
LEARNING_RATE_PRE_TRAIN = 5e-5
LEARNING_RATE_INSTRUCTION_TRAIN = 5e-5
MAX_SEQ_LENGTH = 2048
DATASET_PRE_TRAIN = "dataset_pt.jsonl"
DATASET_INSTRUCTION_TRAIN = "dataset_it.jsonl"
DIRECTORY_PRE_TRAIN = "./pre_train_dir"
DIRECTORY_INSTRUCTION_TRAIN = "./it_train_dir"
EPOCHS_PRE_TRAIN = 3
EPOCHS_INSTRUCTION_TRAIN = 3
CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] + '<|im_end|>\n' }}{% endgeneration %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
LOGGING_STEPS=1