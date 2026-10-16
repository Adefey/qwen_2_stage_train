import logging
import os
import sys
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

from config import *
from objects import *

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            filename=f'logs/log_{datetime.now().strftime("%y_%m_%d_%H-%M-%S")}.log'
        ),
        logging.StreamHandler(stream=sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

logger.info(f"Target device: {device}, dtype: {DATA_TYPE}")

# ----------------- CONTINUED PRE-TRAINING -----------------
logger.info(f"Start Pre-Training of {MODEL_NAME}")

model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DATA_TYPE,
)
if device != "cpu":
    model_base.to(device)

trainer_pre_train = Trainer(
    model=model_base,
    args=training_args_pre_train,
    train_dataset=tokenized_dataset_pre_train,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_pre_train, mlm=False),
)

trainer_pre_train.train()
trainer_pre_train.save_model(DIRECTORY_PRE_TRAIN)
tokenizer_pre_train.save_pretrained(DIRECTORY_PRE_TRAIN)

# ----------------- CHECKPOINT SELECTION -----------------
last_pt_checkpoint = get_last_checkpoint(DIRECTORY_PRE_TRAIN)
pt_model_path = last_pt_checkpoint if last_pt_checkpoint is not None else DIRECTORY_PRE_TRAIN
logger.info(f"Resolved PT checkpoint path: {pt_model_path}")

# ----------------- INSTRUCTION TUNING (SFT) -----------------
logger.info(f"Start Instruction Training from {pt_model_path}")

model_pre_trained = AutoModelForCausalLM.from_pretrained(
    pt_model_path,
    torch_dtype=DATA_TYPE,
)
if device != "cpu":
    model_pre_trained.to(device)

trainer_instruction_train = SFTTrainer(
    model=model_pre_trained,
    args=sft_config,
    train_dataset=raw_dataset_instruction_train,
    processing_class=tokenizer_instruction_train,
)

trainer_instruction_train.train()
trainer_instruction_train.save_model(DIRECTORY_INSTRUCTION_TRAIN)
tokenizer_instruction_train.save_pretrained(DIRECTORY_INSTRUCTION_TRAIN)

# ----------------- INFERENCE CHECK -----------------
logger.info("Evaluating fine-tuned model...")

last_it_checkpoint = get_last_checkpoint(DIRECTORY_INSTRUCTION_TRAIN)
it_model_path = last_it_checkpoint if last_it_checkpoint is not None else DIRECTORY_INSTRUCTION_TRAIN

model_instruction_trained = AutoModelForCausalLM.from_pretrained(
    it_model_path,
    torch_dtype=DATA_TYPE,
)
if device != "cpu":
    model_instruction_trained.to(device)

model_instruction_trained.eval()

test_messages = [{"role": "user", "content": "Кто такой Adefe?"}]

input_ids = tokenizer_instruction_train.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
if device != "cpu":
    input_ids = input_ids.to(device)

with torch.no_grad():
    output_ids = model_instruction_trained.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer_instruction_train.pad_token_id,
    )

response = tokenizer_instruction_train.decode(
    output_ids[0][input_ids.shape[-1]:],
    skip_special_tokens=True,
)

print("\n--- Model Response ---")
print(response)

