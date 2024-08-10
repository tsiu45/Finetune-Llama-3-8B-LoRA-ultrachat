from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

messages = [
    {"role": "system", "content": "Alan Watts's colorful journey explored the depths of philosophy and religion, weaving together Eastern wisdom and Western insights."},
    {"role": "user", "content": "What does the self actually amount to?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
# https://github.com/Lightning-AI/litgpt/issues/327

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response = outputs[0][input_ids.shape[-1]:]

print(tokenizer.decode(response, skip_special_tokens=True))








from transformers import BitsAndBytesConfig

# For 8 bit quantization
#quantization_config = BitsAndBytesConfig(load_in_8bit=True,
#                                        llm_int8_threshold=200.0)

## For 4 bit quantization
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,)

device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             quantization_config=quantization_config, 
                                             device_map=device_map)

model_kwargs = dict(
    attn_implementation="sdpa", #"flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)




from datasets import load_dataset
from datasets import DatasetDict

# based on config
raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")

# remove this when done debugging to include whole dataset
indices = range(0,500)

dataset_dict = {"train": raw_datasets["train_sft"].select(indices),
                "test": raw_datasets["test_sft"].select(indices)}

raw_datasets = DatasetDict(dataset_dict)


example = raw_datasets["train"][0]
messages = example["messages"]
for message in messages:
  role = message["role"]
  content = message["content"]
  print('{0:20}:  {1}'.format(role, content))

import re
import random
from multiprocessing import cpu_count

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 2048



def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

column_names = list(raw_datasets["train"].features)
raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)

# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

for index in random.sample(range(len(raw_datasets["train"])), 3):
  print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")
  print("#####################################")

for index in random.sample(range(len(raw_datasets["test"])), 3):
  print(f"Sample {index} of the processed test set:\n\n{raw_datasets['test'][index]['text']}")
  print("#####################################")








from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
trained_model_id = "Llama-3-8B-sft-lora-ultrachat"
output_dir = '.model_checkpoint/' + trained_model_id

# based on config
training_args = TrainingArguments(
    fp16=False, # specify bf16=True instead when training on GPUs that support bf16 else fp16
    bf16=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1, # originally set to 8
    per_device_train_batch_size=1, # originally set to 8
    push_to_hub=True,
    hub_model_id=trained_model_id,
    # hub_strategy="every_save",
    # report_to="tensorboard",
    report_to="none",  # for skipping wandb logging
    save_strategy="no",
    save_total_limit=None,
    seed=42,
)

# based on config
peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# To clear out cache for unsuccessful run
torch.cuda.empty_cache()

trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=False,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )


train_result = trainer.train()

trainer.push_to_hub()
print('Finished')



