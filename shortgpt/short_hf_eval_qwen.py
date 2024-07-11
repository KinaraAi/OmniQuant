from datasets import load_dataset
import torch,sys,os, random
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
from transformers import default_data_collator, Trainer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
import transformers

# Local imports
from datautils import get_loaders
from utils import *
from preprocessed_data import *
from short_hf_qwen import ShortHFModel

print(sys.argv)


def update_config(path):
    import os, json
    config_path = os.path.join(path, "config.json")
    config_json= {}
    with open(config_path, 'r') as openfile:
        config_json = json.load(openfile)
    config_json["torch_dtype"] = "bfloat16"
    config_json["num_hidden_layers"] = 32 - args.prune_layers
    with open(config_path, 'w') as f:
        json.dump(config_json, f)
        
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--prune_layers", type=int, default=4, help="Number of layers to prune")
parser.add_argument("--seed", type=int, default=10, help="Seed for sampling the calibration data.")
parser.add_argument("--pruned_path", default="./pruned_model", type=str, help="Path to save the pruned model")
parser.add_argument("--lora_trained_path", default="./peftmodel_qwen", type=str, help="Path to save the trained peft model")
parser.add_argument("--merged_path", default="./merged_model", type=str, help="Path to save the merged model")

args = parser.parse_args()


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

        
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        trainloader.append((inp, tar))
    return trainloader, testenc

seed = 10
MAX_SEQ_LEN = 128  
seqlen = MAX_SEQ_LEN

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

batch_size = 16
FFN_MODULES = ["gate_proj","up_proj","down_proj"]#["w1", "w2", "c_proj"]
lora_rank = 64
lora_dropout = 0.05
lora_alpha = lora_rank
warmup_steps = 100
num_training_steps = 1000
scheduler_type="cosine"
peak_lr = 3e-4
samples = 1200
train_samples = 200
num_train_epochs = 4

data = load_dataset('allenai/c4', data_files="multilingual/c4-zh-validation*.json.gz", split = f"train[:{samples}]")
data = data['text']

for index, text in enumerate(data):
    data[index] = trim_string(text, 128)

dataloader = DataLoader(
	data,
	batch_size=batch_size,
	shuffle=True,
	# generator=torch.Generator(device="cuda")
)

# num_train_epochs=num_training_steps // (len(data) // batch_size)
# num_train_epochs = 2

# MAX_SEQ_LEN = 1024
short_model = ShortHFModel(
    model_name= args.model,
    layers_path="model.layers",
    n_prune_layers=args.prune_layers
)
print("Model")
print(short_model.model)
print("*"*50)

print("Layers 0 ")
print(short_model.layers[0])
print("*"*50)

# sample generation
gen = short_model.model.generate(
    short_model.tokenizer(["I am an avid fan of "], return_tensors='pt').input_ids.to("cuda"),
    max_new_tokens=40
)
textgen = short_model.tokenizer.batch_decode(gen, skip_special_tokens=True)

print("generated text: ")
print(textgen)
print("*"*50)

for i, batch in enumerate(tqdm(dataloader)):
    prompts = batch
    short_model.eval_importance(
        prompts=prompts,
        max_seq_len=MAX_SEQ_LEN,
        stride=256,
        max_gen_len=0,
    )
print("Importances")
print(short_model.importances)
print("*"*50)

short_model.remove_layers()
# print("layers to remove")
# print(layers_to_remove)
# print("*"*50)

print("Post Pruning layers")
print(short_model.layers)
print("*"*50)
breakpoint()
# reassign layer_idx to attentions for caching
for layer_idx, module in enumerate(short_model.layers):
    module.self_attn.layer_idx = layer_idx
    
gen = short_model.model.generate(
    short_model.tokenizer(["I am an avid fan of "], return_tensors='pt').input_ids.to("cuda"),
    max_new_tokens=20
)
new_textgen = short_model.tokenizer.batch_decode(gen, skip_special_tokens=True)

print("Post Pruning text:")
print(new_textgen)
print("*"*50)

tokenizer = short_model.tokenizer
model = short_model.model

print("Started Saving Pruned Model")
model.save_pretrained(args.pruned_path)
tokenizer.save_pretrained(args.pruned_path)
update_config(args.pruned_path)
print("Saved Pruned Model")

print("Starting Healing  ... ")

model.train()

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules = FFN_MODULES
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model)

output_dir = "tmp/"

# final_data = data.map(lambda samples: short_model.tokenizer(samples["text"]), batched=True)



# eval_prompt = """
# Summarize this dialog:
# A: Hi Tom, are you busy tomorrow's afternoon?
# B: I'm pretty sure I am. What's up?
# A: Can you go with me to the animal shelter?.
# B: What do you want to do?
# A: I want to get a puppy for my son.
# B: That will make him so happy.
# A: Yeah, we've discussed it many times. I think he's ready now.
# B: That's good. Raising a dog is a tough issue. Like having a baby ;-) 
# A: I'll get him one of those little dogs.
# B: One that won't grow up too big;-)
# A: And eat too much;-))
# B: Do you know which one he would like?
# A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
# B: I bet you had to drag him away.
# A: He wanted to take it home right away ;-).
# B: I wonder what he'll name it.
# A: He said he'd name it after his dead hamster - Lemmy  - he's  a great Motorhead fan :-)))
# ---
# Summary:
# """

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=10)[0], skip_special_tokens=True))
   
train_dataset = get_preprocessed_c4(tokenizer, "train", train_samples, False, 128)
print("Train123:  ", train_dataset)
config = {
    'lora_config': lora_config,
    'learning_rate': peak_lr,
    # 'num_train_epochs': 1,
    'per_device_train_batch_size': batch_size,
    'gradient_checkpointing': False,
}
# num_train_epochs= num_training_steps // (train_samples // batch_size)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    # logging strategies
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused",
    warmup_steps=warmup_steps,
    lr_scheduler_type=scheduler_type,
    num_train_epochs=num_train_epochs ,
    **{k:v for k,v in config.items() if k != 'lora_config'},
    
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
    callbacks=[],
)

# Start training
trainer.train()

model.eval()

print("Saving Trained Model")
model.save_pretrained(args.lora_trained_path, save_adapter=True, save_config=True)
print("Saved Trained Model")

gen = model.generate(
    tokenizer(["I am an avid fan of "], return_tensors='pt').input_ids.to("cuda"),
    max_new_tokens=40
)
new_textgen = tokenizer.batch_decode(gen, skip_special_tokens=True)

print("Post Pruning text:")
print(new_textgen)
print("*"*100)

gen = model.generate(
    tokenizer(["My name is Julien and I like to "], return_tensors='pt').input_ids.to("cuda"),
    max_new_tokens=40
)
new_textgen = tokenizer.batch_decode(gen, skip_special_tokens=True)

print("Post Pruning text:")
print(new_textgen)
print("*"*100)

gen = model.generate(
    tokenizer(["Simply put, the theory of relativity states that "], return_tensors='pt').input_ids.to("cuda"),
    max_new_tokens=40
)
new_textgen = tokenizer.batch_decode(gen, skip_special_tokens=True)

print("Post Pruning text:")
print(new_textgen)
print("*"*100)

gen = model.generate(
    tokenizer(["List all numbers in the Fibonacci sequence: 1, 1, 2, 3, 5, "], return_tensors='pt').input_ids.to("cuda"),
    max_new_tokens=40
)
new_textgen = tokenizer.batch_decode(gen, skip_special_tokens=True)

print("Post Pruning text:")
print(new_textgen)
print("*"*100)

gen = model.generate(
    tokenizer(["What is the value of 1+2+4 ?"], return_tensors='pt').input_ids.to("cuda"),
    max_new_tokens=40
)
new_textgen = tokenizer.batch_decode(gen, skip_special_tokens=True)

print("Post Pruning text:")
print(new_textgen)
print("*"*100)

del model, tokenizer, short_model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.pruned_path), args.lora_trained_path)
tokenizer = AutoTokenizer.from_pretrained(args.pruned_path)
print("Merging the pruned and trained models")
merged_model = model_to_merge.merge_and_unload()

print("Saving the merged model")
merged_model.save_pretrained(args.merged_path)
tokenizer.save_pretrained(args.merged_path)

update_config(args.merged_path)
