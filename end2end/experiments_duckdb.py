import logging
import sys

import datasets
import torch
import transformers
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)

###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir_duckdb",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    "report_to": ["none"],
}

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama-specific layers
)

train_conf = TrainingArguments(**training_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_config}")

################
# Model Loading
################

checkpoint_path = "meta-llama/Llama-3.2-1B-Instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Llama uses eos_token as pad_token
tokenizer.padding_side = 'right'

##################
# Data Processing
##################

def apply_chat_template(
    example,
    tokenizer,
):
    
    messages = []
    user = {"content": f"{example['schema']}\n Input: {example['query']}", "role": "user"}
    assistant = {"content": f"{example['query']}", "role": "assistant"}

    messages = [user, assistant]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

# dataset_name = "motherduckdb/duckdb-text2sql-25k"
# raw_dataset = load_dataset(dataset_name, split="train")
# raw_datasets = raw_dataset.train_test_split(test_size=0.05, seed=42)

# train_dataset = raw_datasets["train"]
# test_dataset = raw_datasets["test"]

train_dataset = Dataset.from_json("train.json")
test_dataset = Dataset.from_json("test.json")
column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to train_sft",
)

processed_test_dataset = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to test_sft",
)


###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_config,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    max_seq_length=2048,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)


# ############
# # Run model inference
# ############

device_map = {"": 0}
new_model = AutoPeftModelForCausalLM.from_pretrained(
    training_config['output_dir'],
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device_map,
)
merged_model = new_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(training_config['output_dir'], trust_remote_code=True)
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
prompt = apply_chat_template(example=test_dataset[0], tokenizer=tokenizer)
outputs = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    num_beams=1,
    temperature=0.3,
    top_k=50,
    top_p=0.95,
    max_time=180,
)
sql = outputs[0]["generated_text"][len(prompt) :].strip()