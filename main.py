import gc
import logging
import os
import sys
from datetime import datetime

import torch
import wandb
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer
from trl import SFTTrainer

from config import DATA_TYPE, DIRECTORY_INSTRUCTION_TRAIN, DIRECTORY_PRE_TRAIN, MODEL_NAME
from objects import (
    device,
    raw_dataset_instruction_eval,
    raw_dataset_instruction_train,
    sft_config,
    tokenized_dataset_pre_train,
    tokenized_eval_datasets_pre_train,
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
        dtype=DATA_TYPE,
    )
    if device != "cpu":
        model_base.to(device)

    trainer_pre_train = Trainer(
        model=model_base,
        args=training_args_pre_train,
        train_dataset=tokenized_dataset_pre_train,
        eval_dataset=tokenized_eval_datasets_pre_train,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_pre_train, mlm=False),
    )

    trainer_pre_train.train()
    trainer_pre_train.save_model(DIRECTORY_PRE_TRAIN)
    tokenizer_pre_train.save_pretrained(DIRECTORY_PRE_TRAIN)

    wandb.finish()

    # fp32 weights + optimizer state of the finished stage would otherwise stay in RAM
    del trainer_pre_train, model_base
    gc.collect()
else:
    logger.info(f"Skipping Pre Training")


# ----------------- INSTRUCTION TUNING (SFT) -----------------
if instruction_train_enabled:
    pt_model_path = DIRECTORY_PRE_TRAIN
    logger.info(f"Start Instruction Training from {pt_model_path}")

    wandb.init(
        project="qwen-trainer",
        name="Instruction Training",
        group=EXPERIMENT_GROUP,
        reinit=True,
    )

    model_pre_trained = AutoModelForCausalLM.from_pretrained(
        pt_model_path,
        dtype=DATA_TYPE,
    )
    if device != "cpu":
        model_pre_trained.to(device)

    trainer_instruction_train = SFTTrainer(
        model=model_pre_trained,
        args=sft_config,
        train_dataset=raw_dataset_instruction_train,
        eval_dataset=raw_dataset_instruction_eval,
        processing_class=tokenizer_instruction_train,
    )

    trainer_instruction_train.train()
    trainer_instruction_train.save_model(DIRECTORY_INSTRUCTION_TRAIN)
    tokenizer_instruction_train.save_pretrained(DIRECTORY_INSTRUCTION_TRAIN)

    wandb.finish()

    del trainer_instruction_train, model_pre_trained
    gc.collect()
else:
    logger.info(f"Skipping Instruction Training")


# ----------------- INFERENCE CHECK -----------------
if inference_enabled:
    # (question, expected answer); questions are deliberately not copied from dataset_it.jsonl
    test_messages = [
        # Facts: paraphrased, reversed and disambiguation questions
        (
            "Как зовут человека с ником Tesen_Desk и где он работает?",
            "Владислав Точигин, Сбер, инженер нагрузочного тестирования",
        ),
        ("Какой ник у Андрея Коленкова?", "Adefe"),
        ("Чья фурсона — голубовато-серая гремучая змея?", "Андрея Коленкова (Adefe), фурсона Amon"),
        (
            "В какой стране стоит сервер Featherweight и какой у него процессор?",
            "Израиль, AMD Ryzen 7 8745H",
        ),
        ("Кто такой ninekeem и из какого он города?", "Александр Якимов из Барнаула, друг Adefe"),
        (
            "Who wrote Libft and what does it do?",
            "Vladislav Tochigin (Tesen_Desk); C printf from scratch on raw syscalls",
        ),
        (
            "Где познакомились Adefe и Tesen_Desk и где была первая встреча?",
            "В боте «Леонардо Дай Винчик», первая встреча в ТЦ «Авиапарк» в Москве",
        ),
        ("When was Vladislav Tochigin born?", "February 21, 1997"),
        ("Adefe работает в Сбере?", "Нет, в NVIDIA (Раанана); в Сбере работает Tesen_Desk"),
        ("Какую музыку крутят на радио Андрея Коленкова?", "EBM, Aggrotech, Nightcore, Industrial"),
        (
            "Who is Dimka's boyfriend and what is his fursona?",
            "Sergey (Chaos / Чаози), black-and-white Sonic-style cat",
        ),
        (
            "Какой Minecraft-сервер основал Adefe и что на нём прошло во время локдауна?",
            "BMSTUCraft, официальный день открытых дверей МГТУ",
        ),
        # Facts that do not exist: the model should not make them up
        ("Какая машина у Adefe?", "Нет информации"),
        ("Сколько детей у Tesen_Desk?", "Нет информации"),
        ("Кто такой Zarkon_42?", "Не знает такого человека"),
        # General topics adjacent to the facts: checks that facts do not leak into them
        ("Какой официальный язык в Израиле?", "Иврит"),
        ("Чем известна компания NVIDIA?", "Видеокарты/GPU, ускорители для ИИ"),
        (
            "Расскажи коротко про город Дмитров.",
            "Город в Московской области, Дмитровский кремль; без Adefe/Tesen_Desk",
        ),
        ("What is Minecraft?", "Sandbox game by Mojang"),
        # General knowledge the model should not forget
        ("Какая столица у Австралии?", "Канберра"),
        ("Кто написал роман «Война и мир»?", "Лев Толстой"),
        ("В каком году человек впервые высадился на Луну?", "1969"),
        ("Why is the sky blue?", "Rayleigh scattering"),
        ("Сколько будет 17 умножить на 23?", "391"),
        ("Напиши функцию на Python, которая проверяет, является ли число простым.", "Рабочий код"),
        (
            "Translate to English: «Кошка спит на подоконнике».",
            "The cat is sleeping on the windowsill",
        ),
    ]

    stop_token_ids = [tokenizer_instruction_train.eos_token_id]
    im_end_id = tokenizer_instruction_train.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id not in stop_token_ids:
        stop_token_ids.append(im_end_id)

    tuned_model_path = os.environ.get("INFERENCE_MODEL_PATH")

    models_eval = {}
    for model_label, model_path in [
        ("base", MODEL_NAME),
        ("pre_trained", DIRECTORY_PRE_TRAIN),
        ("instruction_trained", DIRECTORY_INSTRUCTION_TRAIN),
    ]:
        logger.info(f"Loading {model_label} model from {model_path}")
        models_eval[model_label] = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(device)
        models_eval[model_label].eval()

    for i, (test_message, expected_answer) in enumerate(test_messages):
        model_test_message = [{"role": "user", "content": test_message}]
        inputs = tokenizer_instruction_train.apply_chat_template(
            model_test_message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(device)
        prompt_length = inputs["input_ids"].shape[-1]

        print(f"\n{i + 1} User: {test_message}")
        print(f"{i + 1} Expected: {expected_answer}")

        for model_label, model_eval in models_eval.items():
            with torch.no_grad():
                output_ids = model_eval.generate(
                    **inputs,
                    max_new_tokens=192,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=stop_token_ids,
                    pad_token_id=tokenizer_instruction_train.pad_token_id,
                )

            response = tokenizer_instruction_train.decode(
                output_ids[0][prompt_length:],
                skip_special_tokens=True,
            )
            print(f"{i + 1} [{model_label}]: {response.strip()}")
else:
    logger.info(f"Skipping Inference")


print("Exiting")
