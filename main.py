import logging
import os
import sys
from datetime import datetime

import torch
import wandb
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

from config import DATA_TYPE, DIRECTORY_INSTRUCTION_TRAIN, DIRECTORY_PRE_TRAIN, MODEL_NAME
from objects import (
    device,
    raw_dataset_instruction_train,
    sft_config,
    tokenized_dataset_pre_train,
    tokenizer_instruction_train,
    tokenizer_pre_train,
    training_args_pre_train,
)

EXPERIMENT_GROUP = f"experiment_{datetime.now().strftime('%y_%m_%d_%H-%M-%S')}"
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

# PRE_TRAIN, INSTRUCTION_TRAIN, INFERENCE
pre_train_enabled_str = os.environ.get("PRE_TRAIN", "0")
pre_train_enabled = pre_train_enabled_str.strip().lower() in ("yes", "true", "t", "1")

instruction_train_enabled_str = os.environ.get("INSTRUCTION_TRAIN", "0")
instruction_train_enabled = instruction_train_enabled_str.strip().lower() in (
    "yes",
    "true",
    "t",
    "1",
)

inference_enabled_str = os.environ.get("INFERENCE", "0")
inference_enabled = inference_enabled_str.strip().lower() in ("yes", "true", "t", "1")

logger.info(
    f"Selected stages: {pre_train_enabled=} {instruction_train_enabled=} {inference_enabled=}"
)
logger.info(f"Target device: {device}, dtype: {DATA_TYPE}")

# ----------------- CONTINUED PRE-TRAINING -----------------
if pre_train_enabled:
    logger.info(f"Start Pre Training of {MODEL_NAME}")

    wandb.init(
        project="qwen-trainer",
        name="Pre Training",
        group=EXPERIMENT_GROUP,
        reinit=True,
    )

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

    wandb.finish()
else:
    logger.info(f"Skipping Pre Training")

# ----------------- INSTRUCTION TUNING (SFT) -----------------
if instruction_train_enabled:
    last_pt_checkpoint = get_last_checkpoint(DIRECTORY_PRE_TRAIN)
    pt_model_path = last_pt_checkpoint if last_pt_checkpoint is not None else DIRECTORY_PRE_TRAIN
    logger.info(f"Resolved Pre Training checkpoint path: {pt_model_path}")

    logger.info(f"Start Instruction Training from {pt_model_path}")

    wandb.init(
        project="qwen-trainer",
        name="Instruction Training",
        group=EXPERIMENT_GROUP,
        reinit=True,
    )

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

    wandb.finish()
else:
    logger.info(f"Skipping Instruction Training")

# ----------------- INFERENCE CHECK -----------------
if inference_enabled:
    logger.info("Evaluating fine-tuned model...")

    last_it_checkpoint = get_last_checkpoint(DIRECTORY_INSTRUCTION_TRAIN)
    it_model_path = (
        last_it_checkpoint if last_it_checkpoint is not None else DIRECTORY_INSTRUCTION_TRAIN
    )

    model_instruction_trained = AutoModelForCausalLM.from_pretrained(
        it_model_path,
        dtype=DATA_TYPE,
    )
    if device != "cpu":
        model_instruction_trained.to(device)

    model_instruction_trained.eval()

    test_messages = [
        "Кто такой Adefe?",
        "Кто такой Tesen_Desk?",
        "Какая столица у Франции?",
        "Какой язык используется в Израиле?",
        "Кто такой Adefe?",
        "Где работает Tesen_Desk и чем он там занимается?",
        "Что такое Amon и как он связан с никнеймом «Змейка»?",
        "Кто такой ninekeem и откуда он?",
        "Что такое Libft и почему это важно для Tesen_Desk?",
        "Какие проекты хостятся на сервере Featherweight и где он физически находится?",
        "What is the difference between `git merge` and `git rebase`?",
        "What does a load balancer do, and what are common algorithms it uses?",
        "Explain in simple words what a Docker container is.",
        "Чем отличаются TCP и UDP и когда что использовать?",
    ]

    for i, test_message in enumerate(test_messages):

        model_test_message = [{"role": "user", "content": test_message}]
        inputs = tokenizer_instruction_train.apply_chat_template(
            model_test_message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(device if device != "cpu" else "cpu")

        stop_token_ids = [tokenizer_instruction_train.eos_token_id]
        im_end_id = tokenizer_instruction_train.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in stop_token_ids:
            stop_token_ids.append(im_end_id)

        with torch.no_grad():
            output_ids = model_instruction_trained.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer_instruction_train.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        response = tokenizer_instruction_train.decode(
            output_ids[0][prompt_length:],
            skip_special_tokens=True,
        )

        print(f"{i + 1} User: {test_message}")
        print(f"{i + 1} Model: {response.strip()}")
else:
    logger.info(f"Skipping Inference")

print("Exiting")
