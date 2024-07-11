import torch
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import random
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import random, torch

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def unpack_signed_int32_to_int4(number):
	# Mask to extract 4 bits at a time
	mask = 0b1111
	# List to store unpacked 4-bit integers
	unpacked_int4 = []

	# Loop through 8 segments of 4 bits
	for _ in range(8):
		# Extract the least significant 4 bits
		segment = number & mask
		# If the 4th bit is 1, it's a negative number, so convert to signed
		if segment & 0b1000:
			segment = -((~segment & mask) + 1)
		# Otherwise, it's positive, so keep as is
		else:
			segment = segment & mask
		# Append to the list
		unpacked_int4.append(segment)
		# Shift the number to the right by 4 bits for the next segment
		number >>= 4

	# Reverse the list to maintain the correct order
	unpacked_int4.reverse()
	return unpacked_int4
# unpacked_qweight = unpack_signed_int32_to_int4(1669888392)
# unpacked_qzero = unpack_signed_int32_to_int4(2004318071)
# print("qweight: ", unpacked_qweight)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name of model path")
parser.add_argument("--gptq_model", default="Qwen/Qwen1.5-7B-Chat-GPTQ-Int4", type=str, help="model path for the gptq model")
parser.add_argument("--output_dir", default="./merged_gptq_weights", type=str, help="Path to save the merged model")
parser.add_argument("--before_eval_ppl",action = "store_true")
parser.add_argument("--after_eval_ppl",action = "store_true")

args = parser.parse_args()
import os
def create_folder(folder_name):
	current_path = os.getcwd()
	folder_path = os.path.join(current_path, folder_name)

	if not os.path.exists(folder_path):
		os.makedirs(folder_path, exist_ok=True)  # exist_ok avoids errors if it already exists
		print(f"Folder '{folder_name}' created successfully!")
	else:
		print(f"Folder '{folder_name}' already exists.")
	
def dequantization(qweight, qzeros, scales, g_idx,bits=4, group_size=128):
	# Create a tensor for bitwise right shift operation
	wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0).to(qzeros.device)
	# Apply bitwise right shift and convert qzeros to the appropriate type
	zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int16 if bits == 8 else torch.int8)
	torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros) #zeros (86,512,8)
	# Reshape the zeros tensor
	zeros = zeros + 1
	zeros = zeros.reshape(scales.shape) # zeros (86,1,4096)

	# Reshape the scales tensor
	# scales = scales.reshape(-1, 1, scales.shape[-1]) # scales (86,1,4096)

	# Similar bitwise right shift operation for qweight and reshape
	# breakpoint()
	weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8) # weight (1376,8,4096)
	torch.bitwise_and(weight, (2 ** bits) - 1, out=weight) # weight (1376,8,4096)
	# weight = weight.reshape(-1, group_size, weight.shape[2]) # weight (86,128,4096)

	# Apply dequantization formula and reshape the final weight
	
	weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
	# breakpoint()
	weight = scales[g_idx] * (weight - zeros[g_idx])
	# Return the transposed weight
	return weight.transpose(0, 1)

import transformers

@torch.no_grad()
def evaluate(model):
	# model.to('cuda')
	tokenizer = transformers.AutoTokenizer.from_pretrained(
		pretrained_model_name_or_path=args.model,
		model_max_length=2048,
		padding_side="right",
		use_fast=False,
	)
	tokenizer.pad_token = tokenizer.eos_token


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
			tar[:, :-1] = -100
			trainloader.append((inp, tar))
		return trainloader, testenc

	nsamples = 128
	seqlen = 2048
	seed = 0

	dataloader, testloader = get_wikitext2(
		nsamples = nsamples,
		seed = seed,
		seqlen = seqlen,
		tokenizer = tokenizer,
	)
	
	testenc = testloader.input_ids

	# use_cache = model.config.use_cache
	model.config.use_cache = False
	model.eval()
	nlls = []
	nsamples = testenc.numel() // seqlen
	for i in tqdm(range(nsamples)):
		with torch.no_grad():
			batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
			logits = model(batch)['logits'].to('cpu')
			# batch.to('cpu')
			shift_logits = logits[:, :-1, :]
			shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(
				shift_logits.view(-1, shift_logits.size(-1)),
				shift_labels.view(-1),
			)
			neg_log_likelihood = loss.float() * seqlen
			nlls.append(neg_log_likelihood)
			del batch, logits, shift_logits, shift_labels
			torch.cuda.empty_cache()

	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
	print("perplexity: ", ppl.item())


create_folder(args.output_dir)
# model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
# if args.before_eval_ppl:
# 	print("Calculating perplexity before replacing weights...")
# 	evaluate(model)
	
# del tokenizer
# exit()
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", trust_remote_code=True)
gptq_model = AutoModelForCausalLM.from_pretrained(args.gptq_model, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)                                
from tqdm import tqdm
for i in tqdm(range(32)):
	scales = gptq_model.model.layers[i].mlp.down_proj.scales
	qweight = gptq_model.model.layers[i].mlp.down_proj.qweight
	qzeros = gptq_model.model.layers[i].mlp.down_proj.qzeros
	g_idx = gptq_model.model.layers[i].mlp.down_proj.g_idx
	dq_weights = dequantization(qweight, qzeros, scales, g_idx,4)
	model.model.layers[i].mlp.down_proj.weight = torch.nn.Parameter(dq_weights.detach().cpu().contiguous())
	torch.cuda.empty_cache()
del gptq_model
model.to(torch.bfloat16)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
torch.cuda.empty_cache()

# model.to('cuda')
if args.after_eval_ppl:
	print("Calculating perplexity after replacing weights...")
	evaluate(model)
