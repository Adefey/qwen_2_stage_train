import logging
import sys
from datetime import datetime

from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer
from trl import SFTTrainer

from config import *
from objects import *

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

model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DATA_TYPE, device_map=device
)

logger.info(f"Device: {device}")

logger.info(f"Start Pre Training of {MODEL_NAME}")

trainer_pre_train = Trainer(
    model=model_base,
    args=training_args_pre_train,
    train_dataset=tokenized_dataset_pre_train,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_pre_train, mlm=False),
)

trainer_pre_train.train()
trainer_pre_train.save_model(DIRECTORY_PRE_TRAIN)

logger.info(f"Start Instruction Training of {MODEL_NAME}")

model_pre_trained = AutoModelForCausalLM.from_pretrained(
    DIRECTORY_PRE_TRAIN, torch_dtype=DATA_TYPE, device_map=device
)

trainer_instruction_train = SFTTrainer(
    model=model_pre_trained,
    args=sft_config,
    train_dataset=tokenized_dataset_instruction_train,
)

trainer_instruction_train.train()
trainer_instruction_train.save_model(DIRECTORY_INSTRUCTION_TRAIN)

logger.info(f"Start Instruction Training of {MODEL_NAME}")

model_instruction_trained = AutoModelForCausalLM.from_pretrained(
    DIRECTORY_INSTRUCTION_TRAIN, torch_dtype=DATA_TYPE, device_map=device
)
model_instruction_trained.eval()

test_messages = [{"role": "user", "content": "Кто такой Adefe?"}]

input_ids = tokenizer_instruction_train.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model_instruction_trained.device)

with torch.no_grad():
    output_ids = model_instruction_trained.generate(
        input_ids,
        max_new_tokens=MAX_SEQ_LENGTH,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=model_instruction_trained.eos_token_id,
    )

responses = [
    tokenizer_instruction_train.decode(
        output_ids[i][input_ids.shape[-1] :],
        skip_special_tokens=True,
    )
    for i in range(len(test_messages))
]

print(responses)
